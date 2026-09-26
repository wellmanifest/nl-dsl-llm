// Taskand NL-DSL-LLM Client Adapter
// Connects Taskand dev/chat to the shared local daemon via Unix Domain Socket or TCP
// Complies with wellmanifest/nl-dsl-llm standard v0.2.0

import net from 'node:net';

export class TaskandNLClient {
  constructor(options = {}) {
    this.socketPath = options.socketPath || process.env.NL_DSL_SOCKET || '/run/paxlet/nl-dsl-llm.sock';
    this.host = options.host || process.env.NL_DSL_HOST || '127.0.0.1';
    this.port = options.port || Number(process.env.NL_DSL_PORT || 50051);
    this.timeoutMs = options.timeoutMs || 5000;
  }

  /**
   * Interpret natural language query into validated procedure URI and parameters
   * @param {string} query Natural language command (PL / EN / DE / UK)
   * @param {object} [context] Context options (session_id, locale, allow_llm)
   * @returns {Promise<{uri: string, actionId: string, arguments: object, latencyMs: number, sourceLayer: string}>}
   */
  async interpret(query, context = {}) {
    const payload = {
      action: 'interpret',
      query: String(query).trim(),
      client_id: 'taskand',
      locale: context.locale || 'pl',
      session_id: context.sessionId || 'default',
      allow_llm_fallback: context.allowLlm ?? true,
      model_target: context.modelTarget || 'qwen_2b'
    };

    return new Promise((resolve, reject) => {
      const socket = net.createConnection(this.socketPath);
      let buffer = '';
      const timer = setTimeout(() => {
        socket.destroy();
        reject(new Error(`Timeout (${this.timeoutMs}ms) connecting to ${this.socketPath}`));
      }, this.timeoutMs);

      socket.on('connect', () => {
        socket.write(JSON.stringify(payload) + '\n');
      });

      socket.on('data', chunk => {
        buffer += chunk.toString('utf8');
        if (buffer.includes('\n')) {
          socket.end();
        }
      });

      socket.on('end', () => {
        clearTimeout(timer);
        try {
          const res = JSON.parse(buffer.trim());
          resolve({
            status: res.status || 'OK',
            actionId: res.action_id || '',
            uri: res.resolved_procedure_uri || '',
            arguments: res.arguments || {},
            sourceLayer: res.matched_layer || 'unknown',
            latencyMs: res.meta?.total_latency_ms || 0
          });
        } catch (e) {
          reject(new Error(`Invalid JSON response: ${buffer}`));
        }
      });

      socket.on('error', err => {
        clearTimeout(timer);
        // Fallback for standalone development when daemon is not currently running
        resolve(this._inMemoryFallback(query));
      });
    });
  }

  /**
   * Fast in-process fallback when daemon socket is unavailable
   */
  _inMemoryFallback(query) {
    const low = query.toLowerCase();
    if (low.includes('cpu') || low.includes('procesor')) {
      return {
        status: 'MATCHED',
        actionId: 'monitor.cpu_status',
        uri: 'proc://taskand.dev/monitor/cpu/v1',
        arguments: {},
        sourceLayer: 'in_process_fallback',
        latencyMs: 0.1
      };
    }
    return {
      status: 'UNSUPPORTED',
      actionId: '',
      uri: '',
      arguments: {},
      sourceLayer: 'in_process_fallback',
      latencyMs: 0.1
    };
  }
}

export default TaskandNLClient;
