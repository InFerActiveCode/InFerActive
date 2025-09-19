import React, { useEffect, useRef } from 'react';
import styled from 'styled-components';

// Styled components
const ExplorerContainer = styled.div`
  position: absolute;
  background: white;
  border: 1px solid #ddd;
  border-radius: 4px;
  pointer-events: auto;
  font-size: 12px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
  z-index: 1500; /* Higher z-index to display above other panels */
  max-width: 250px;
  min-width: 180px;
  max-height: 300px;
  display: flex;
  flex-direction: column;
  overflow: hidden;
`;

const TokensList = styled.div`
  overflow-y: auto;
  max-height: 250px;
  display: flex;
  flex-direction: column;
`;

const TokenItem = styled.div<{ isSelected: boolean }>`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 12px;
  cursor: pointer;
  background: ${props => props.isSelected ? '#e3f2fd' : 'white'};
  border-left: 3px solid ${props => props.isSelected ? '#1976d2' : 'transparent'};
  transition: all 0.2s ease;
  
  &:not(:last-child) {
    border-bottom: 1px solid #f0f0f0;
  }
  
  &:hover {
    background: ${props => props.isSelected ? '#bbdefb' : '#f8f9fa'};
  }
  
  .token-text {
    flex-grow: 1;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    font-weight: ${props => props.isSelected ? 'bold' : 'normal'};
  }
  
  .token-prob {
    margin-left: 8px;
    color: #666;
    font-size: 11px;
  }
`;

// Type definitions
interface TokenInfo {
  token: string;
  prob: number;
  nodeId: string;
}

interface TokenExplorerProps {
  position: { x: number, y: number };
  containerRef: React.RefObject<HTMLDivElement | null>;
  currentToken: TokenInfo;
  alternativeTokens: TokenInfo[];
  onTokenSelect: (nodeId: string) => void;
  onClose: () => void;
  isVisible: boolean;
}

export const TokenExplorer: React.FC<TokenExplorerProps> = ({
  position,
  containerRef,
  currentToken,
  alternativeTokens,
  onTokenSelect,
  onClose,
  isVisible
}) => {
  const explorerRef = useRef<HTMLDivElement>(null);
  
  // Position calculation and adjustment
  useEffect(() => {
    if (!isVisible || !explorerRef.current || !containerRef.current) return;
    
    const explorer = explorerRef.current;
    const container = containerRef.current;
    const containerRect = container.getBoundingClientRect();
    
    // Set default position below the token
    let left = position.x;
    let top = position.y + 20; // Position below token
    
    // Check if explorer exceeds container right boundary
    const explorerWidth = explorer.offsetWidth;
    if (left + explorerWidth > containerRect.width) {
      left = Math.max(0, containerRect.width - explorerWidth - 10); // 10px right margin
    }
    
    // Apply position
    explorer.style.left = `${left}px`;
    explorer.style.top = `${top}px`;
  }, [position, containerRef, isVisible]);
  
  if (!isVisible) return null;
  
  return (
    <ExplorerContainer ref={explorerRef}>
      <TokensList>
        {alternativeTokens.map((token, index) => {
          const isSelected = token.token === currentToken.token && token.prob === currentToken.prob;
          
          return (
            <TokenItem
              key={`${token.nodeId}-${index}`}
              isSelected={isSelected}
              onClick={() => {
                // Ignore if already selected
                if (isSelected) {
                  return;
                }
                onTokenSelect(token.nodeId);
              }}
            >
              <span className="token-text">{token.token}</span>
              <span className="token-prob">{(token.prob * 100).toFixed(1)}%</span>
            </TokenItem>
          );
        })}
        
        {alternativeTokens.length === 0 && (
          <div style={{ padding: '12px', color: '#666', textAlign: 'center' }}>
No alternative tokens available.
          </div>
        )}
      </TokensList>
    </ExplorerContainer>
  );
};

export default TokenExplorer;