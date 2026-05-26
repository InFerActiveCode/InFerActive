import React from 'react';
import styled from 'styled-components';
import { CompletedSequence, TokenSelection } from '../types/types';

const EOT_TOKENS = new Set(['<|eot_id|>', '</s>', '<|endoftext|>']);

const Container = styled.div`
  height: 100%;
  overflow: hidden;
  display: flex;
  flex-direction: column;
`;

const StreamSection = styled.div`
  padding: 12px 14px;
  display: flex;
  flex-direction: column;
  gap: 10px;
  background: var(--genesis-surface);
  border: 1px solid var(--genesis-border);
  border-radius: var(--genesis-radius-md);
  flex: 1;
  min-height: 0;
  overflow: hidden;
`;

const Header = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--genesis-space-2);
`;

const Title = styled.h2`
  margin: 0;
  display: flex;
  align-items: center;
  gap: var(--genesis-space-2);
  color: var(--genesis-text-primary);
  font-size: 14px;
  font-weight: 700;
  line-height: 1;
`;

const StreamViewport = styled.div<{ $nodeSelected: boolean }>`
  flex: 1;
  min-height: 0;
  overflow: auto;
  padding: 12px 14px;
  border-radius: var(--genesis-radius-sm);
  border: 1px solid ${props => props.$nodeSelected ? 'var(--genesis-primary)' : 'var(--genesis-border)'};
  background: var(--genesis-surface);
`;

const OutputText = styled.div`
  color: var(--genesis-text-primary);
  font-size: 15px;
  line-height: 1.72;
  white-space: pre-wrap;
  word-break: break-word;
  user-select: text;
`;

const ActionRow = styled.div`
  display: flex;
  gap: 10px;
  padding: 0 2px;
`;

const EvaluationButton = styled.button<{ $active: boolean; $variant: 'good' | 'bad' }>`
  flex: 1;
  min-height: 38px;
  padding: 8px 12px;
  font-size: 12px;
  font-weight: 700;
  line-height: 1;
  border-radius: var(--genesis-radius-sm);
  cursor: pointer;
  border: 1px solid ${props => {
    if (props.$active) return props.$variant === 'good' ? 'var(--genesis-success)' : 'var(--genesis-error)';
    return 'var(--genesis-border)';
  }};
  background: ${props => {
    if (props.$active) return props.$variant === 'good' ? 'var(--genesis-success)' : 'var(--genesis-error)';
    return 'var(--genesis-surface)';
  }};
  color: ${props => {
    if (props.$active) return '#fff';
    return props.$variant === 'good' ? 'var(--genesis-success)' : 'var(--genesis-error)';
  }};
  transition: background 0.15s, border-color 0.15s, color 0.15s;

  &:hover {
    border-color: ${props => props.$variant === 'good' ? 'var(--genesis-success)' : 'var(--genesis-error)'};
    background: ${props => {
      if (props.$active) return props.$variant === 'good' ? '#43a047' : '#e53935';
      return props.$variant === 'good' ? '#f0fdf4' : '#fef2f2';
    }};
  }

  &:focus-visible {
    outline: none;
    box-shadow: var(--genesis-focus-ring);
  }
`;

const Placeholder = styled.div`
  margin: auto;
  text-align: center;
  color: var(--genesis-text-secondary);
  font-size: 13px;
  line-height: 1.6;
  padding: 24px 16px;
  border: 1px dashed var(--genesis-border);
  border-radius: var(--genesis-radius-sm);
  background: var(--genesis-background);
  max-width: 520px;
`;

interface TokenStreamPanelProps {
  completedSequence: CompletedSequence | null;
  selectedToken?: TokenSelection;
  onTokenSelect?: (sequenceId: string, tokenIndex: number) => void;
  alternatives?: Array<{ token: string; probability: number; nodeId: string }>;
  completedSequences?: CompletedSequence[];
  onSequenceSelect?: (sequence: CompletedSequence) => void;
  isGoodMarked?: boolean;
  isNodeSelected?: boolean;
  isBadMarked?: boolean;
  onMarkGood?: () => void;
  onMarkBad?: () => void;
  onExpandNode?: () => void;
}

const getViewportBorderColor = (nodeSelected: boolean, goodMarked: boolean, badMarked: boolean): string => {
  if (goodMarked) return '#4caf50';
  if (badMarked) return '#ef4444';
  if (nodeSelected) return '#6366f1';
  return '#e9ecef';
};

const getOutputText = (completedSequence: CompletedSequence): string => {
  const visibleTokens = completedSequence.tokens.filter(token => !EOT_TOKENS.has(token.token));
  if (visibleTokens.length > 0) {
    return visibleTokens.map(token => token.token).join('');
  }

  return Array.from(EOT_TOKENS).reduce(
    (text, marker) => text.split(marker).join(''),
    completedSequence.text
  );
};

export const TokenStreamPanel: React.FC<TokenStreamPanelProps> = ({
  completedSequence,
  isGoodMarked = false,
  isNodeSelected = false,
  isBadMarked = false,
  onMarkGood,
  onMarkBad,
  onExpandNode,
}) => {
  if (!completedSequence) {
    return (
      <Container>
        <StreamSection>
          <Header>
            <Title>Response</Title>
          </Header>
          <StreamViewport
            $nodeSelected={false}
            style={{ borderColor: getViewportBorderColor(false, false, false) }}
          >
            <Placeholder>
              Select a token node from the tree or a sequence from the output panel.
            </Placeholder>
          </StreamViewport>
        </StreamSection>
      </Container>
    );
  }

  const outputText = getOutputText(completedSequence);

  return (
    <Container>
      <StreamSection>
        <Header>
          <Title>Response</Title>
        </Header>
        <StreamViewport
          $nodeSelected={isNodeSelected}
          style={{ borderColor: getViewportBorderColor(isNodeSelected, isGoodMarked, isBadMarked) }}
        >
          <OutputText>{outputText || ' '}</OutputText>
        </StreamViewport>
        <ActionRow>
          <EvaluationButton type="button" onClick={onMarkGood} $active={isGoodMarked} $variant="good" title="Mark as good">
            Mark Good
          </EvaluationButton>
          <EvaluationButton type="button" onClick={onMarkBad} $active={isBadMarked} $variant="bad" title="Mark as bad">
            Mark Bad
          </EvaluationButton>
          {onExpandNode && (
            <EvaluationButton type="button" onClick={onExpandNode} $active={false} $variant="good" title="Expand selected node">
              Expand
            </EvaluationButton>
          )}
        </ActionRow>
      </StreamSection>
    </Container>
  );
};
