import { InferenceRequest, InferenceMode, ModelControlRequest, VisualNode } from '../types/types';

export class TokenGenerationService {
  static createModelLoadRequest(requestId?: string): ModelControlRequest {
    return {
      type: 'load_model',
      ...(requestId ? { request_id: requestId } : {})
    };
  }

  static createModelStatusCheckRequest(requestId?: string): ModelControlRequest {
    return {
      type: 'check_model_status',
      ...(requestId ? { request_id: requestId } : {})
    };
  }


  static createInferenceRequest(
    mode: InferenceMode,
    inputText: string,
    params: {
      k: number;
      temperature: number;
      topP: number;
      minP: number;
      maxTokens?: number;
      depth?: number;
      requestId?: string;
    }
  ): InferenceRequest {
    const { k, temperature, topP, minP, maxTokens, depth, requestId } = params;

    this.validateInferenceParams({ k, temperature, topP, minP });
    const isTreeInitialization = mode === 'generate_with_bfs' || mode === 'generate_with_smc';

    const request: InferenceRequest = {
      type: mode,
      input_text: inputText,
      k,
      temperature,
      top_p: topP,
      min_p: minP,
      ...(isTreeInitialization
        ? { depth, max_tokens: maxTokens }
        : { depth }),
      ...(requestId ? { request_id: requestId } : {})
    };

    return request;
  }


  static createNodeExplorationRequest(
    inputText: string,
    nodeId: string,
    params: {
      k: number;
      temperature: number;
      topP: number;
      minP: number;
      maxTokens: number;
      requestId?: string;
    }
  ): InferenceRequest {
    const { k, temperature, topP, minP, maxTokens, requestId } = params;

    this.validateInferenceParams({ k, temperature, topP, minP });

    return {
      type: 'explore_node',
      input_text: inputText,
      k,
      temperature,
      top_p: topP,
      min_p: minP,
      max_tokens: maxTokens,
      node_id: nodeId,
      ...(requestId ? { request_id: requestId } : {})
    };
  }

  static validateInferenceParams(params: {
    k: number;
    temperature: number;
    topP: number;
    minP: number;
  }): void {
    const { k, temperature, topP, minP } = params;

    if (k < 1) {
      throw new Error('k should be at least 1.');
    }

    if (temperature < 0 || temperature > 2) {
      throw new Error('temperature should be between 0 and 2.');
    }

    if (topP < 0 || topP > 1) {
      throw new Error('top_p should be between 0 and 1.');
    }

    if (minP < 0 || minP > 1) {
      throw new Error('min_p should be between 0 and 1.');
    }
  }


  static getDefaultParameters() {
    return {
      inputText: "Give me the very short, one sentence poem. Return the sentence only.",
      k: 5,
      temperature: 0.7,
      topP: 0.9,
      minP: 0.05,
      maxTokens: 20,
      depth: 3
    };
  }

  static extractParametersFromMessage(message: Partial<InferenceRequest> | null | undefined) {
    if (!message) return null;

    return {
      inputText: message.input_text ?? "",
      k: message.k ?? 5,
      temperature: message.temperature ?? 0.7,
      topP: message.top_p ?? 0.9,
      minP: message.min_p ?? 0.05,
      maxTokens: message.max_tokens ?? 20,
      depth: message.depth ?? 3
    };
  }

  static isNodeGenerating(node: VisualNode | null): boolean {
    if (!node) return false;
    return node.nodeState === 'generating';
  }

  static isNodeCompleted(node: VisualNode | null): boolean {
    if (!node) return false;
    return node.nodeState === 'completed';
  }

  static canGenerateFromNode(
    node: VisualNode | null,
    timeoutMs: number = 30000
  ): boolean {
    if (!node) return false;

    if (this.isNodeCompleted(node)) return false;

    if (this.isNodeGenerating(node) && node.lastGenerationRequestTime) {
      const elapsedTime = Date.now() - node.lastGenerationRequestTime;
      if (elapsedTime < timeoutMs) return false;
    }

    return true;
  }
}
