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
   * Real-time autocomplete suggestions and option network DAG
   * @param {string} partialQuery Partial typing or streaming speech token (e.g. "zre")
   * @param {object} [twinContext] Digital Twin environment context
   * @returns {Promise<{success: boolean, suggestions: Array<object>, latencyMs: number}>}
   */
  async suggest(partialQuery, twinContext = {}) {
    const payload = {
      action: 'suggest',
      partial_query: String(partialQuery).trim(),
      client_id: 'taskand',
      twin_context: {
        environment: twinContext.environment || 'prod',
        active_services: twinContext.activeServices || ['postgres', 'redis-cache'],
        degraded_services: twinContext.degradedServices || ['redis-cache'],
        active_tickets: twinContext.activeTickets || ['ticket-001', 'ticket-002'],
        cluster_nodes: twinContext.clusterNodes || []
      }
    };

    return new Promise(resolve => {
      const socket = net.createConnection(this.socketPath);
      let buffer = '';
      const timer = setTimeout(() => {
        socket.destroy();
        resolve(this._inMemorySuggestFallback(partialQuery, twinContext));
      }, 500);

      socket.on('connect', () => {
        socket.write(JSON.stringify(payload) + '\n');
      });

      socket.on('data', chunk => {
        buffer += chunk.toString('utf8');
        if (buffer.includes('\n')) socket.end();
      });

      socket.on('end', () => {
        clearTimeout(timer);
        try {
          const res = JSON.parse(buffer.trim());
          resolve(res);
        } catch {
          resolve(this._inMemorySuggestFallback(partialQuery, twinContext));
        }
      });

      socket.on('error', () => {
        clearTimeout(timer);
        resolve(this._inMemorySuggestFallback(partialQuery, twinContext));
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

  /**
   * Fast in-process option network generator from Digital Twin
   */
  _inMemorySuggestFallback(partialQuery, twinContext = {}) {
    const clean = String(partialQuery).toLowerCase().trim();
    const env = twinContext.environment || 'prod';
    const services = twinContext.activeServices || ['postgres', 'redis-cache'];
    const degraded = twinContext.degradedServices || ['redis-cache'];
    const suggestions = [];

    if ('zrestartuj'.startsWith(clean) || clean.startsWith('zre') || clean.startsWith('rest')) {
      for (const svc of services) {
        const isDegraded = degraded.includes(svc);
        suggestions.push({
          id: `service.restart:${svc}`,
          display_label: `Zrestartuj ${svc}${isDegraded ? ' (Zdegradowany)' : ` [${env}]`}`,
          completion_text: `zrestartuj ${svc}`,
          action_id: 'service.restart',
          resolved_uri: `proc://taskand.dev/service/restart/v1`,
          bound_arguments: { target: svc, env },
          confidence: isDegraded ? 0.98 : 0.90,
          requires_confirmation: env === 'prod',
          badge: isDegraded ? 'DEGRADED' : 'PROD',
          next_options: [
            { id: `${svc}:graceful`, display_label: 'Łagodny restart', completion_text: `zrestartuj ${svc} --graceful` },
            { id: `${svc}:force`, display_label: 'Wymuszony restart', completion_text: `zrestartuj ${svc} --force` }
          ]
        });
      }
    }
    return {
      success: true,
      partial_query: partialQuery,
      suggestions: suggestions.sort((a, b) => (b.badge === 'DEGRADED' ? 1 : -1)),
      latency_ms: 0.3
    };
  }

  /**
   * Route an API Menu click or form submission to canonical proc:// procedure
   * @param {string} endpointId e.g. "service.restart" or "monitor.cpu"
   * @param {object} formValues e.g. { target: "redis-cache", env: "prod" }
   * @param {object} viewContext e.g. { currentRoute: "/dashboard", activeService: "redis-cache" }
   */
  routeApiMenuAction(endpointId, formValues = {}, viewContext = {}) {
    const action = endpointId.startsWith('proc://')
      ? endpointId
      : `proc://taskand.dev/${endpointId.replace(/\./g, '/')}/v1`;
    const target = formValues.target || viewContext.activeService || 'default';
    const env = formValues.env || formValues.environment || 'prod';
    return {
      success: true,
      resolved_uri: action,
      bound_arguments: { ...formValues, target, env },
      requires_confirmation: env === 'prod',
      latency_ms: 0.1
    };
  }
}

export default TaskandNLClient;
