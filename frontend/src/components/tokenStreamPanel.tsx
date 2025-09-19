import React, { useState, useRef, useEffect } from 'react';
import styled from 'styled-components';
import { CompletedSequence, TokenInfo, TokenSelection } from '../types/types';
import TokenExplorer from './tokenExplorer';

const Container = styled.div`
  height: 100%;
  overflow: hidden;
`;

// Output Panel Section
const OutputSection = styled.div`
  padding: 16px;
  overflow-y: auto;
  background: white;
`;

// Token Stream Section
const StreamSection = styled.div`
  padding: 12px;
  overflow: auto;
  display: flex;
  flex-direction: column;
  background: white;
  border: 1px solid #e9ecef;
  border-radius: 8px;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
`;

const AddButton = styled.button`
  width: 28px;
  height: 28px;
  border-radius: 50%;
  border: none;
  background: #007bff;
  color: white;
  font-size: 18px;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  transition: background 0.2s;
  margin-left: 8px;
  flex-shrink: 0;

  &:hover {
    background: #0056b3;
  }

  &:disabled {
    background: #dee2e6;
    cursor: not-allowed;
  }
`;

// Output Panel styles
const SequenceList = styled.div`
  display: flex;
  flex-direction: column;
  gap: 8px;
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

const TokenStreamWrapper = styled.div`
  display: flex;
  align-items: center;
  padding: 12px;
  background: #fafbfc;
  border-radius: 6px;
  min-height: 50px;
  position: relative;
  border: 1px solid #e9ecef;
`;

const TokenContent = styled.div`
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  line-height: 1.6;
  flex: 1;
  padding-right: 8px;
`;

const TokenContainer = styled.span`
  position: relative;
  margin: 0;
  padding: 0;
  display: inline;
`;

const Token = styled.span<{ 
  probability: number; 
  isSelected: boolean; 
  isHovered: boolean;
}>`
  display: inline;
  padding: ${props => (props.isHovered || props.isSelected) ? '2px 4px' : '0'};
  margin: ${props => (props.isHovered || props.isSelected) ? '0 1px' : '0'};
  background-color: ${props => {
    if (props.isSelected) return '#e3f2fd';
    if (props.isHovered) return '#f8f9fa';
    return 'transparent';
  }};
  border: ${props => {
    if (props.isSelected) return '1px solid #2196f3';
    if (props.isHovered) return '1px solid #dee2e6';
    return '1px solid transparent';
  }};
  border-radius: 4px;
  cursor: pointer;
  font-family: inherit;
  position: relative;
  transition: all 0.2s ease;
  white-space: ${props => (props.isHovered || props.isSelected) ? 'normal' : 'pre'};
`;

const ProbabilityLabel = styled.div<{ isVisible: boolean }>`
  font-size: 10px;
  color: #666;
  text-align: center;
  margin-top: 2px;
  opacity: ${props => props.isVisible ? 1 : 0};
  height: ${props => props.isVisible ? 'auto' : '0'};
  overflow: hidden;
  transition: opacity 0.2s ease, height 0.2s ease;
  position: absolute;
  top: 100%;
  left: 50%;
  transform: translateX(-50%);
  background: white;
  padding: 1px 4px;
  border-radius: 2px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
  white-space: nowrap;
`;

const Placeholder = styled.div`
  text-align: center;
  color: #6c757d;
  font-size: 14px;
`;

interface TokenStreamPanelProps {
  completedSequence: CompletedSequence | null;
  selectedToken: TokenSelection;
  onTokenSelect: (sequenceId: string, tokenIndex: number) => void;
  alternatives?: Array<{ token: string; probability: number; nodeId: string }>;
  completedSequences?: CompletedSequence[];
  onSequenceSelect?: (sequence: CompletedSequence) => void;
}

export const TokenStreamPanel: React.FC<TokenStreamPanelProps> = ({
  completedSequence,
  selectedToken,
  onTokenSelect,
  alternatives = [],
  completedSequences = [],
  onSequenceSelect
}) => {
  // Track hovered token index
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  // Tooltip position state
  const [tooltipPosition, setTooltipPosition] = useState({ x: 0, y: 0 });
  // Tooltip visibility
  const [showTooltip, setShowTooltip] = useState(false);
  // Store currently selected token
  const [currentTokenNodeId, setCurrentTokenNodeId] = useState<string | null>(null);
  
  // Container and token element refs
  const wrapperRef = useRef<HTMLDivElement>(null);
  const tokenRefs = useRef<(HTMLSpanElement | null)[]>([]);
  
  // Update current token ID when selected token changes
  useEffect(() => {
    if (completedSequence && selectedToken.sequenceId === completedSequence.id && 
        selectedToken.tokenIndex !== null && 
        selectedToken.tokenIndex < completedSequence.tokens.length) {
      const tokenInfo = completedSequence.tokens[selectedToken.tokenIndex];
      setCurrentTokenNodeId(tokenInfo.nodeId);
    } else {
      setCurrentTokenNodeId(null);
    }
  }, [selectedToken, completedSequence]);
  
  // Convert current token info for TokenExplorer
  const getCurrentTokenInfo = () => {
    if (!completedSequence || selectedToken.tokenIndex === null) return null;
    
    const token = completedSequence.tokens[selectedToken.tokenIndex];
    return {
      token: token.token,
      prob: token.prob,
      nodeId: token.nodeId
    };
  };
  
  // Convert alternative token info for TokenExplorer
  const getAlternativeTokensInfo = () => {
    return alternatives.map(alt => ({
      token: alt.token,
      prob: alt.probability,
      nodeId: alt.nodeId
    }));
  };
  
  // Click event handler
  const handleTokenClick = (sequenceId: string, index: number, event: React.MouseEvent) => {
    // Calculate token position
    const tokenElement = tokenRefs.current[index];
    if (tokenElement && wrapperRef.current) {
      const tokenRect = tokenElement.getBoundingClientRect();
      const wrapperRect = wrapperRef.current.getBoundingClientRect();
      
      // Pass token bottom position clearly
      setTooltipPosition({
        x: tokenRect.left - wrapperRect.left,
        y: tokenRect.bottom - wrapperRect.top
      });
    }

    // Select different token
    onTokenSelect(sequenceId, index);
    setShowTooltip(true);
    
    event.stopPropagation(); // Prevent event bubbling
  };
  
  // Alternative token selection handler
  const handleAlternativeTokenSelect = (nodeId: string) => {
    
    // For completed nodes, only attempt to find sequence
    const sequence = completedSequence;
    if (sequence) {
      const targetTokenIndex = sequence.tokens.findIndex(t => t.nodeId === nodeId);
      if (targetTokenIndex >= 0) {
        onTokenSelect(sequence.id, targetTokenIndex);
      }
    }
  };
  
  // Close tooltip on outside click
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (wrapperRef.current && !wrapperRef.current.contains(event.target as Node)) {
        setShowTooltip(false);
      }
    };
    
    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);
  
  // Function to render each token
  const renderTokens = () => {
    if (!completedSequence) {
      return (
        <TokenContent>
          <Placeholder>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;No sequence selected. Choose a sequence from the output panel or a token node from the tree.
          </Placeholder>
        </TokenContent>
      );
    }

    // Initialize token ref array
    tokenRefs.current = completedSequence.tokens.map((_, i) => tokenRefs.current[i] || null);
  
    return (
      <TokenContent style={{ fontFamily: 'system-ui, -apple-system, sans-serif', fontSize: '15px' }}>
        {completedSequence.tokens.map((token, index) => {

          const isSelected = 
            selectedToken.sequenceId === completedSequence.id && 
            selectedToken.tokenIndex === index;
          
          const isHovered = hoveredIndex === index;

          // Handle special characters (spaces, tabs, etc.)
          let displayToken = token.token;
          
          // Display while preserving leading/trailing spaces
          if (displayToken === ' ') {
            displayToken = ' '; // Preserve space
          }

          return (
            <TokenContainer key={`${completedSequence.id}-${index}`}>
              <Token 
                ref={(el) => { tokenRefs.current[index] = el; }}
                probability={token.prob}
                isSelected={isSelected}
                isHovered={isHovered}
                onMouseEnter={() => setHoveredIndex(index)}
                onMouseLeave={() => setHoveredIndex(null)}
              >
                {displayToken}
                {(isHovered || isSelected) && (
                  <ProbabilityLabel isVisible={isHovered || isSelected}>
                    {(token.prob * 100).toFixed(1)}%
                  </ProbabilityLabel>
                )}
              </Token>
            </TokenContainer>
          );
        })}
      </TokenContent>
    );
  };

  // Render TokenExplorer
  const renderTokenExplorer = () => {
    const currentToken = getCurrentTokenInfo();
    if (!currentToken || !showTooltip) return null;
    
    const alternatives = getAlternativeTokensInfo();
    
    return (
      <TokenExplorer
        position={tooltipPosition}
        containerRef={wrapperRef}
        currentToken={currentToken}
        alternativeTokens={alternatives}
        onTokenSelect={handleAlternativeTokenSelect}
        onClose={() => setShowTooltip(false)}
        isVisible={showTooltip}
      />
    );
  };

  // Extract text excluding EOT tokens from sequence
  const getDisplayText = (sequence: CompletedSequence): string => {
    const eotPatterns = ['<|eot_id|>', '</s>', '<|endoftext|>'];
    return sequence.tokens
      .filter(token => !eotPatterns.includes(token.token))
      .map(token => token.token)
      .join('');
  };

  // Sequence click handler
  const handleSequenceClick = (sequence: CompletedSequence) => {
    if (onSequenceSelect) {
      onSequenceSelect(sequence);
    }
  };

  return (
    <Container>
      <StreamSection>
        <TokenStreamWrapper ref={wrapperRef} onClick={() => setShowTooltip(false)}>
          {renderTokens()}
          {renderTokenExplorer()}
        </TokenStreamWrapper>
      </StreamSection>
    </Container>
  );
};