import React from 'react';
import styled from 'styled-components';
import { CompletedSequence, TokenSelection } from '../types/types';

const Container = styled.div`
  padding: 16px;
  border: 1px solid #e9ecef;
  border-radius: 8px;
  background: white;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
  height: 100%;
  overflow: auto;
  display: flex;
  flex-direction: column;
`;

const Title = styled.h2`
  margin: 0 0 16px 0;
  color: #343a40;
  font-size: 18px;
  font-weight: 600;
`;

const SequenceList = styled.div`
  display: flex;
  flex-direction: column;
  gap: 8px;
  overflow-y: auto;
`;

const SequenceItem = styled.div<{ isSelected: boolean }>`
  padding: 10px;
  background: ${props => props.isSelected ? '#e3f2fd' : 'white'};
  border: 1px solid ${props => props.isSelected ? '#2196f3' : '#dee2e6'};
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.2s ease;
  font-size: 13px;

  &:hover {
    border-color: ${props => props.isSelected ? '#1976d2' : '#adb5bd'};
    background: ${props => props.isSelected ? '#e3f2fd' : '#f8f9fa'};
  }
`;

const SequenceText = styled.div`
  margin-top: 6px;
  font-family: system-ui, -apple-system, sans-serif;
  font-size: 13px;
  line-height: 1.4;
  white-space: pre-wrap;
  word-break: break-word;
  max-height: 60px;
  overflow: hidden;
  text-overflow: ellipsis;
`;

const ProbabilityBadge = styled.span`
  float: right;
  padding: 2px 6px;
  background: #e9ecef;
  border-radius: 4px;
  font-size: 11px;
  color: #495057;
`;

interface OutputPanelProps {
  completedSequences: CompletedSequence[];
  selectedSequence: TokenSelection;
  onSequenceSelect: (sequence: CompletedSequence) => void;
  onTokenSelect: (sequenceId: string, tokenIndex: number) => void;
}

export const OutputPanel: React.FC<OutputPanelProps> = ({
  completedSequences,
  selectedSequence,
  onSequenceSelect,
  onTokenSelect,
}) => {
  // Extract text from sequence excluding EOT tokens
  const getDisplayText = (sequence: CompletedSequence): string => {
    const eotPatterns = ['<|eot_id|>', '</s>', '<|endoftext|>'];

    return sequence.tokens
      .map(token => token.token)
      .join('');
  };

  const handleSequenceClick = (sequence: CompletedSequence) => {
    onSequenceSelect(sequence);
  };

  return (
    <Container>
      <Title>Generated Sequences</Title>
      <SequenceList>
        {completedSequences.map((sequence) => (
          <SequenceItem
            key={sequence.id}
            isSelected={selectedSequence.sequenceId === sequence.id}
            onClick={() => handleSequenceClick(sequence)}
          >
            <ProbabilityBadge>
              {(sequence.totalProb * 100).toFixed(2)}%
            </ProbabilityBadge>
            <SequenceText>
              {getDisplayText(sequence)}
            </SequenceText>
          </SequenceItem>
        ))}
        {completedSequences.length === 0 && (
          <div style={{ 
            textAlign: 'center', 
            color: '#6c757d', 
            padding: '30px',
            fontSize: '13px'
          }}>
            No completed sequences yet. Start inference to see results.
          </div>
        )}
      </SequenceList>
    </Container>
  );
};

export default OutputPanel;