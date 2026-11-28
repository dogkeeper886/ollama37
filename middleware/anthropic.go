package middleware

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/anthropic"
	"github.com/ollama/ollama/api"
)

// A streamed request that has had no response for startDelay, while the model loads
// or reads the prompt, gets message_start and then a ping every pingInterval, so the
// client sees the stream alive (#608). A quicker failure keeps its HTTP status.
var (
	startDelay   = 5 * time.Second
	pingInterval = 10 * time.Second
)

// AnthropicWriter wraps the response writer to transform Ollama responses to Anthropic format
type AnthropicWriter struct {
	BaseWriter
	stream    bool
	id        string
	model     string
	converter *anthropic.StreamConverter

	// mu serializes the handler's writes with keepAlive's.
	mu sync.Mutex
	// started: message_start went out with status 200 before the handler responded.
	started bool
	// responded: the handler has written its first response; keepAlive stops.
	responded bool
	// status is what the handler asked for after started, when 200 was already sent.
	status int
}

// keepAlive sends message_start after startDelay without a response, then pings,
// until the handler responds or done closes.
func (w *AnthropicWriter) keepAlive(done <-chan struct{}) {
	wait := startDelay
	for {
		select {
		case <-done:
			return
		case <-time.After(wait):
		}

		w.mu.Lock()
		// A handler that chose a failure status answers with it.
		if w.responded || (w.status != 0 && w.status != http.StatusOK) {
			w.mu.Unlock()
			return
		}
		events := []anthropic.StreamEvent{{Event: "ping", Data: anthropic.PingEvent{Type: "ping"}}}
		if !w.started {
			w.started = true
			w.ResponseWriter.WriteHeader(http.StatusOK)
			events = w.converter.Start()
		}
		for _, e := range events {
			if err := w.writeEvent(e.Event, e.Data); err != nil {
				w.mu.Unlock()
				return
			}
		}
		w.mu.Unlock()
		wait = pingInterval
	}
}

// WriteHeader keeps the status the handler asks for, so that once 200 is sent
// Write can report a failure as an error event.
func (w *AnthropicWriter) WriteHeader(code int) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.status = code
	if !w.started {
		w.ResponseWriter.WriteHeader(code)
	}
}

func (w *AnthropicWriter) Flush() {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.ResponseWriter.Flush()
}

// writeStreamError sends a failure that came after message_start as an error event.
func (w *AnthropicWriter) writeStreamError(data []byte) (int, error) {
	var errData struct {
		Error string `json:"error"`
	}
	if err := json.Unmarshal(data, &errData); err != nil {
		return 0, err
	}
	if err := w.writeEvent("error", anthropic.NewError(w.status, errData.Error)); err != nil {
		return 0, err
	}
	return len(data), nil
}

func (w *AnthropicWriter) writeError(data []byte) (int, error) {
	var errData struct {
		Error string `json:"error"`
	}
	if err := json.Unmarshal(data, &errData); err != nil {
		return 0, err
	}

	w.ResponseWriter.Header().Set("Content-Type", "application/json")
	err := json.NewEncoder(w.ResponseWriter).Encode(anthropic.NewError(w.ResponseWriter.Status(), errData.Error))
	if err != nil {
		return 0, err
	}

	return len(data), nil
}

func (w *AnthropicWriter) writeEvent(eventType string, data any) error {
	d, err := json.Marshal(data)
	if err != nil {
		return err
	}
	_, err = w.ResponseWriter.Write([]byte(fmt.Sprintf("event: %s\ndata: %s\n\n", eventType, d)))
	if err != nil {
		return err
	}
	if f, ok := w.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
	return nil
}

func (w *AnthropicWriter) writeResponse(data []byte) (int, error) {
	var chatResponse api.ChatResponse
	err := json.Unmarshal(data, &chatResponse)
	if err != nil {
		return 0, err
	}

	if w.stream {
		w.ResponseWriter.Header().Set("Content-Type", "text/event-stream")

		events := w.converter.Process(chatResponse)
		for _, event := range events {
			if err := w.writeEvent(event.Event, event.Data); err != nil {
				return 0, err
			}
		}
		return len(data), nil
	}

	w.ResponseWriter.Header().Set("Content-Type", "application/json")
	response := anthropic.ToMessagesResponse(w.id, chatResponse)
	return len(data), json.NewEncoder(w.ResponseWriter).Encode(response)
}

func (w *AnthropicWriter) Write(data []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.responded = true

	if w.started && w.status != 0 && w.status != http.StatusOK {
		return w.writeStreamError(data)
	}

	code := w.ResponseWriter.Status()
	if code != http.StatusOK {
		return w.writeError(data)
	}

	return w.writeResponse(data)
}

// AnthropicMessagesMiddleware handles Anthropic Messages API requests
func AnthropicMessagesMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		var req anthropic.MessagesRequest
		err := c.ShouldBindJSON(&req)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, anthropic.NewError(http.StatusBadRequest, err.Error()))
			return
		}

		if req.Model == "" {
			c.AbortWithStatusJSON(http.StatusBadRequest, anthropic.NewError(http.StatusBadRequest, "model is required"))
			return
		}

		if req.MaxTokens <= 0 {
			c.AbortWithStatusJSON(http.StatusBadRequest, anthropic.NewError(http.StatusBadRequest, "max_tokens is required and must be positive"))
			return
		}

		if len(req.Messages) == 0 {
			c.AbortWithStatusJSON(http.StatusBadRequest, anthropic.NewError(http.StatusBadRequest, "messages is required"))
			return
		}

		chatReq, err := anthropic.FromMessagesRequest(req)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, anthropic.NewError(http.StatusBadRequest, err.Error()))
			return
		}

		var b bytes.Buffer
		if err := json.NewEncoder(&b).Encode(chatReq); err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, anthropic.NewError(http.StatusInternalServerError, err.Error()))
			return
		}

		c.Request.Body = io.NopCloser(&b)

		messageID := anthropic.GenerateMessageID()

		w := &AnthropicWriter{
			BaseWriter: BaseWriter{ResponseWriter: c.Writer},
			stream:     req.Stream,
			id:         messageID,
			model:      req.Model,
			converter:  anthropic.NewStreamConverter(messageID, req.Model),
		}

		if req.Stream {
			c.Writer.Header().Set("Content-Type", "text/event-stream")
			c.Writer.Header().Set("Cache-Control", "no-cache")
			c.Writer.Header().Set("Connection", "keep-alive")
		}

		c.Writer = w

		if !req.Stream {
			c.Next()
			return
		}

		done := make(chan struct{})
		var wg sync.WaitGroup
		wg.Add(1)
		go func() {
			defer wg.Done()
			w.keepAlive(done)
		}()
		c.Next()
		close(done)
		wg.Wait()
	}
}
