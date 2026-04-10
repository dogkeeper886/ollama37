/**
 * LogCollector - Captures docker compose logs with text markers for precise
 * test boundary extraction.
 *
 * Design:
 * - Spawns `docker compose logs --follow --timestamps` as background process
 * - Writes all logs to persistent session file: /tmp/ollama37-session-{timestamp}.log
 * - Injects text markers for test start/end boundaries
 * - Extracts per-test logs using sed
 *
 * Marker format: ===TEST:{TEST_ID}:{START|END}:{ISO_TIMESTAMP}===
 */

import { spawn, ChildProcess, execSync } from 'child_process';
import {
  createWriteStream,
  WriteStream,
  writeFileSync,
  readFileSync,
  existsSync,
  unlinkSync,
  readdirSync,
  mkdirSync,
} from 'fs';
import path from 'path';

interface TestMarkerState {
  startWritten: boolean;
  endWritten: boolean;
  startTime: string;
  endTime: string;
}

export class LogCollector {
  private process: ChildProcess | null = null;
  private sessionFile: string;
  private logFileStream: WriteStream | null = null;
  private dockerComposeDir: string;
  private isRunning: boolean = false;
  private testMarkers: Map<string, TestMarkerState> = new Map();
  private writeQueue: Promise<void> = Promise.resolve();
  private lineBuffer: string = '';
  private outputDir: string;

  constructor(dockerComposeDir: string, outputDir: string) {
    this.dockerComposeDir = dockerComposeDir;
    this.outputDir = outputDir;
    this.sessionFile = `/tmp/ollama37-session-${Date.now()}.log`;
  }

  /**
   * Start the log collector background process.
   */
  async start(): Promise<void> {
    if (this.isRunning) {
      return;
    }

    // Clean up old session files (> 24 hours)
    this.cleanupOldSessions();

    return new Promise((resolve, reject) => {
      // Create file stream
      this.logFileStream = createWriteStream(this.sessionFile, { flags: 'a' });

      // Write session header
      this.logFileStream.write(
        `===SESSION:START:${new Date().toISOString()}===\n`
      );

      // Spawn docker compose logs --follow
      this.process = spawn(
        'docker',
        ['compose', 'logs', '--follow', '--timestamps'],
        {
          cwd: this.dockerComposeDir,
          stdio: ['ignore', 'pipe', 'pipe'],
        }
      );

      this.isRunning = true;

      // Pipe stdout to file with line buffering
      this.process.stdout?.on('data', (data: Buffer) => {
        this.processLogData(data, false);
      });

      // Pipe stderr to file with prefix
      this.process.stderr?.on('data', (data: Buffer) => {
        this.processLogData(data, true);
      });

      this.process.on('error', (err) => {
        this.isRunning = false;
        reject(err);
      });

      this.process.on('close', () => {
        this.isRunning = false;
      });

      // Give docker compose time to start
      setTimeout(() => {
        if (this.isRunning) {
          resolve();
        } else {
          reject(new Error('Log collector failed to start'));
        }
      }, 500);
    });
  }

  /**
   * Stop the log collector background process.
   */
  async stop(): Promise<void> {
    if (!this.process || !this.isRunning) {
      return;
    }

    // Flush any remaining buffered data
    if (this.lineBuffer) {
      this.queueWrite(this.lineBuffer + '\n');
      this.lineBuffer = '';
    }

    return new Promise((resolve) => {
      let resolved = false;
      const doResolve = () => {
        if (!resolved) {
          resolved = true;
          this.isRunning = false;

          // Write session end marker and close stream
          if (this.logFileStream && !this.logFileStream.destroyed) {
            this.logFileStream.write(
              `===SESSION:END:${new Date().toISOString()}===\n`
            );
            this.logFileStream.end();
          }

          resolve();
        }
      };

      this.process!.on('close', doResolve);
      this.process!.kill('SIGTERM');

      // Force kill after 5 seconds
      setTimeout(() => {
        if (this.isRunning && this.process) {
          this.process.kill('SIGKILL');
        }
        doResolve();
      }, 5000);
    });
  }

  /**
   * Reconnect the log stream after a container restart.
   * Kills the old process and spawns a new one, appending to the same session file.
   */
  async reconnect(): Promise<void> {
    if (this.process) {
      this.process.kill('SIGTERM');
      this.process = null;
      this.isRunning = false;
    }

    // Flush remaining buffer
    if (this.lineBuffer) {
      this.queueWrite(this.lineBuffer + '\n');
      this.lineBuffer = '';
    }

    return new Promise((resolve, reject) => {
      this.process = spawn(
        'docker',
        ['compose', 'logs', '--follow', '--timestamps', '--since', '1s'],
        {
          cwd: this.dockerComposeDir,
          stdio: ['ignore', 'pipe', 'pipe'],
        }
      );

      this.isRunning = true;

      this.process.stdout?.on('data', (data: Buffer) => {
        this.processLogData(data, false);
      });

      this.process.stderr?.on('data', (data: Buffer) => {
        this.processLogData(data, true);
      });

      this.process.on('error', (err) => {
        this.isRunning = false;
        reject(err);
      });

      this.process.on('close', () => {
        this.isRunning = false;
      });

      setTimeout(() => {
        if (this.isRunning) {
          resolve();
        } else {
          reject(new Error('Log collector failed to reconnect'));
        }
      }, 500);
    });
  }

  /**
   * Mark the start of a test.
   */
  markTestStart(testId: string): void {
    // Clean up any stale test log file
    const testLogPath = this.getTestLogPath(testId);
    if (existsSync(testLogPath)) {
      try {
        unlinkSync(testLogPath);
      } catch {
        // Ignore cleanup errors
      }
    }

    const timestamp = new Date().toISOString();
    this.testMarkers.set(testId, { startWritten: true, endWritten: false, startTime: timestamp, endTime: '' });
    this.queueWrite(`===TEST:${testId}:START:${timestamp}===\n`);
  }

  /**
   * Mark the end of a test.
   */
  markTestEnd(testId: string): void {
    const timestamp = new Date().toISOString();
    const state = this.testMarkers.get(testId);
    if (state) {
      state.endWritten = true;
      state.endTime = timestamp;
    }
    this.queueWrite(`===TEST:${testId}:END:${timestamp}===\n`);
  }

  /**
   * Extract and save logs for a specific test.
   * Returns the path to the saved log file.
   */
  extractTestLogs(testId: string): string {
    const outputPath = this.getTestLogPath(testId);

    // Ensure output directory exists
    const dir = path.dirname(outputPath);
    if (!existsSync(dir)) {
      mkdirSync(dir, { recursive: true });
    }

    // Wait for pending writes
    this.syncFlush();

    const markerState = this.testMarkers.get(testId);
    if (!markerState?.startWritten) {
      writeFileSync(outputPath, '');
      return outputPath;
    }

    try {
      const escapedTestId = this.escapeRegex(testId);
      let sedCmd: string;

      if (markerState.endWritten) {
        // Extract between START and END markers
        sedCmd = `sed -n '/===TEST:${escapedTestId}:START:/,/===TEST:${escapedTestId}:END:/{/===TEST:/d;p}' "${this.sessionFile}"`;
      } else {
        // Extract from START to EOF (test still running)
        sedCmd = `sed -n '/===TEST:${escapedTestId}:START:/,${'$'}{/===TEST:/d;p}' "${this.sessionFile}"`;
      }

      const result = execSync(sedCmd, {
        encoding: 'utf-8',
        maxBuffer: 10 * 1024 * 1024,
      });

      // Strip ANSI codes to reduce log size
      let cleaned = this.stripAnsi(result);

      // Fallback: if session stream had no data (e.g. container restarted),
      // query docker logs directly using the stored timestamps.
      if (cleaned.trim().length === 0 && markerState.startTime) {
        cleaned = this.fallbackDockerLogs(markerState.startTime, markerState.endTime);
      }

      writeFileSync(outputPath, cleaned);
    } catch {
      // Even on sed failure, try fallback
      try {
        if (markerState.startTime) {
          const fallback = this.fallbackDockerLogs(markerState.startTime, markerState.endTime);
          writeFileSync(outputPath, fallback);
          return outputPath;
        }
      } catch {
        // Ignore fallback errors
      }
      writeFileSync(outputPath, '');
    }

    return outputPath;
  }

  /**
   * Get logs for a specific test as a string.
   */
  getLogsForTest(testId: string): string {
    const logPath = this.extractTestLogs(testId);
    try {
      return readFileSync(logPath, 'utf-8');
    } catch {
      return '';
    }
  }

  /**
   * Get all logs collected so far.
   */
  getAllLogs(): string {
    this.syncFlush();
    try {
      return readFileSync(this.sessionFile, 'utf-8');
    } catch {
      return '';
    }
  }

  /**
   * Check if the collector is running.
   */
  isActive(): boolean {
    return this.isRunning;
  }

  /**
   * Get the session file path.
   */
  getSessionFilePath(): string {
    return this.sessionFile;
  }

  /**
   * Copy session file to output directory.
   */
  copySessionToOutput(): string {
    const outputPath = path.join(this.outputDir, 'session.log');
    if (!existsSync(this.outputDir)) {
      mkdirSync(this.outputDir, { recursive: true });
    }
    try {
      const content = readFileSync(this.sessionFile, 'utf-8');
      writeFileSync(outputPath, content);
    } catch {
      writeFileSync(outputPath, '');
    }
    return outputPath;
  }

  // ============================================
  // Private methods
  // ============================================

  private getTestLogPath(testId: string): string {
    return path.join(this.outputDir, `${testId}.log`);
  }

  private processLogData(data: Buffer, isStderr: boolean): void {
    const text = this.lineBuffer + data.toString();
    const lines = text.split('\n');

    // Keep incomplete last line in buffer
    this.lineBuffer = lines.pop() || '';

    // Write complete lines
    if (lines.length > 0) {
      const prefix = isStderr ? '[stderr] ' : '';
      const formatted = lines.map((l) => prefix + l).join('\n') + '\n';
      this.queueWrite(formatted);
    }
  }

  private queueWrite(data: string): void {
    this.writeQueue = this.writeQueue.then(() => {
      return new Promise<void>((resolve) => {
        if (this.logFileStream && !this.logFileStream.destroyed) {
          this.logFileStream.write(data, () => resolve());
        } else {
          resolve();
        }
      });
    });
  }

  private syncFlush(): void {
    if (this.logFileStream && !this.logFileStream.destroyed) {
      const fd = (this.logFileStream as unknown as { fd?: number }).fd;
      if (typeof fd === 'number') {
        try {
          const { fsyncSync } = require('fs');
          fsyncSync(fd);
        } catch {
          // Ignore fsync errors
        }
      }
    }
  }

  private fallbackDockerLogs(sinceTime: string, untilTime: string): string {
    try {
      const args = ['logs', 'ollama37', '--since', sinceTime];
      if (untilTime) {
        args.push('--until', untilTime);
      }
      const result = execSync(`docker ${args.join(' ')}`, {
        encoding: 'utf-8',
        maxBuffer: 10 * 1024 * 1024,
      });
      return this.stripAnsi(result);
    } catch {
      return '';
    }
  }

  private escapeRegex(str: string): string {
    return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }

  private stripAnsi(str: string): string {
    // eslint-disable-next-line no-control-regex
    return str.replace(/\x1b\[[0-9;]*m/g, '');
  }

  private cleanupOldSessions(): void {
    const oneDay = 24 * 60 * 60 * 1000;
    const now = Date.now();

    try {
      const files = readdirSync('/tmp').filter((f) =>
        f.startsWith('ollama37-session-')
      );
      for (const file of files) {
        const match = file.match(/ollama37-session-(\d+)\.log/);
        if (match) {
          const timestamp = parseInt(match[1]);
          if (now - timestamp > oneDay) {
            try {
              unlinkSync(`/tmp/${file}`);
            } catch {
              // Ignore individual cleanup errors
            }
          }
        }
      }
    } catch {
      // Ignore cleanup errors
    }
  }
}
