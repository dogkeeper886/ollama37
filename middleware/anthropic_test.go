package middleware

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
)

// sseEvents returns the event names of an SSE body in order, and the data of each error event.
func sseEvents(t *testing.T, body string) (names []string, errors []map[string]any) {
	t.Helper()
	for _, block := range strings.Split(strings.TrimSpace(body), "\n\n") {
		var name, data string
		for _, line := range strings.Split(block, "\n") {
			if v, ok := strings.CutPrefix(line, "event: "); ok {
				name = v
			} else if v, ok := strings.CutPrefix(line, "data: "); ok {
				data = v
			}
		}
		names = append(names, name)
		if name == "error" {
			var e map[string]any
			if err := json.Unmarshal([]byte(data), &e); err != nil {
				t.Fatalf("error event data %q: %v", data, err)
			}
			errors = append(errors, e)
		}
	}
	return names, errors
}

// TestAnthropicKeepAlive covers #608: a streamed request that waits on the model gets
// message_start and pings, while a quick answer or a quick failure stays as before.
func TestAnthropicKeepAlive(t *testing.T) {
	gin.SetMode(gin.TestMode)
	defer func(d, p time.Duration) { startDelay, pingInterval = d, p }(startDelay, pingInterval)
	startDelay, pingInterval = 30*time.Millisecond, 30*time.Millisecond

	answer := func(c *gin.Context) {
		for _, r := range []api.ChatResponse{
			{Message: api.Message{Role: "assistant", Content: "Hi."}},
			{Done: true, DoneReason: "stop"},
		} {
			b, _ := json.Marshal(r)
			c.Writer.Write(b)
			c.Writer.Flush()
		}
	}
	fail := func(code int) gin.HandlerFunc {
		return func(c *gin.Context) { c.JSON(code, gin.H{"error": "boom"}) }
	}
	slow := func(h gin.HandlerFunc) gin.HandlerFunc {
		return func(c *gin.Context) { time.Sleep(150 * time.Millisecond); h(c) }
	}
	answerEvents := []string{"content_block_start", "content_block_delta", "content_block_stop", "message_delta", "message_stop"}

	cases := []struct {
		name    string
		stream  bool
		handler gin.HandlerFunc
		status  int
		// events, with "ping+" for one or more pings; nil for a JSON body
		events  []string
		errType string
	}{
		{"quick answer", true, answer, http.StatusOK, append([]string{"message_start"}, answerEvents...), ""},
		{"slow answer", true, slow(answer), http.StatusOK, append([]string{"message_start", "ping+"}, answerEvents...), ""},
		{"quick failure", true, fail(http.StatusNotFound), http.StatusNotFound, nil, ""},
		{"slow failure", true, slow(fail(http.StatusInternalServerError)), http.StatusOK, []string{"message_start", "ping+", "error"}, "api_error"},
		{"slow answer, not streamed", false, slow(answer), http.StatusOK, nil, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			router := gin.New()
			router.POST("/v1/messages", AnthropicMessagesMiddleware(), tc.handler)
			body := `{"model":"m","max_tokens":8,"stream":` + map[bool]string{true: "true", false: "false"}[tc.stream] +
				`,"messages":[{"role":"user","content":"hi"}]}`
			req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
			resp := httptest.NewRecorder()
			router.ServeHTTP(resp, req)

			if resp.Code != tc.status {
				t.Fatalf("status = %d, want %d; body %s", resp.Code, tc.status, resp.Body)
			}
			if tc.events == nil {
				if strings.Contains(resp.Body.String(), "event: ") {
					t.Fatalf("want a JSON body, got events: %s", resp.Body)
				}
				return
			}
			got, errs := sseEvents(t, resp.Body.String())
			var collapsed []string
			for _, e := range got {
				if e == "ping" {
					if n := len(collapsed); n > 0 && collapsed[n-1] == "ping+" {
						continue
					}
					e = "ping+"
				}
				collapsed = append(collapsed, e)
			}
			if strings.Join(collapsed, " ") != strings.Join(tc.events, " ") {
				t.Fatalf("events = %v, want %v", got, tc.events)
			}
			if tc.errType != "" {
				if e, _ := errs[0]["error"].(map[string]any); e["type"] != tc.errType || e["message"] != "boom" {
					t.Fatalf("error event = %v, want type %s, message boom", errs[0], tc.errType)
				}
			}
		})
	}
}
