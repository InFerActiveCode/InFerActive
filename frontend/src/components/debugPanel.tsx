import React, { useRef, useEffect, useState } from 'react';
import styled from 'styled-components';
import { VisualNode } from '../types/types';
import { 
  calculateProbabilityCoverage, 
  calculateDepthWiseCoverage 
} from '../utils/treeTransform';

const DebugContainer = styled.div`
  position: fixed;
  top: 70px;
  left: 500px;
  width: 400px;
  max-height: 500px;
  background: white;
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  display: flex;
  flex-direction: column;
  z-index: 1000;
  border: 1px solid #dee2e6;
  overflow: hidden;
`;

const DebugHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 12px;
  background: #f8f9fa;
  border-bottom: 1px solid #e9ecef;
`;

const HeaderTitle = styled.h3`
  margin: 0;
  font-size: 14px;
  font-weight: 600;
  color: #343a40;
`;

const CloseButton = styled.button`
  background: none;
  border: none;
  color: #6c757d;
  font-size: 16px;
  cursor: pointer;
  padding: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 24px;
  height: 24px;
  border-radius: 4px;
  
  &:hover {
    background: #e9ecef;
    color: #343a40;
  }
`;

const StatusBar = styled.div<{ connected: boolean }>`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 12px;
  font-size: 12px;
  background-color: ${props => props.connected ? '#d4edda' : '#f8d7da'};
  color: ${props => props.connected ? '#155724' : '#721c24'};
`;

const MessagesContainer = styled.div`
  flex: 1;
  overflow-y: auto;
  padding: 8px;
  max-height: 320px;
  font-family: monospace;
  font-size: 12px;
`;

const Message = styled.div<{ direction: 'sent' | 'received'; isCollapsed: boolean }>`
  margin: 4px 0;
  padding: 6px 8px;
  background-color: ${props => props.direction === 'sent' ? '#e9ecef' : '#f8f9fa'};
  border: 1px solid ${props => props.direction === 'sent' ? '#dee2e6' : '#e9ecef'};
  border-radius: 4px;
  position: relative;
  word-wrap: break-word;
  white-space: pre-wrap;
  
  .message-toggle {
    cursor: pointer;
    margin-left: 4px;
    opacity: 0.7;
    transition: opacity 0.2s;
    
    &:hover {
      opacity: 1;
    }
  }
`;

const MessageHeader = styled.div`
  display: flex;
  justify-content: space-between;
  margin-bottom: 4px;
  font-size: 11px;
  color: #6c757d;
`;

const MessageDirection = styled.span<{ direction: 'sent' | 'received' }>`
  font-weight: bold;
  color: ${props => props.direction === 'sent' ? '#007bff' : '#28a745'};
`;

const MessageContent = styled.pre<{ isCollapsed: boolean }>`
  margin: 0;
  font-family: monospace;
  font-size: 11px;
  white-space: pre-wrap;
  word-break: break-word;
  max-height: ${props => props.isCollapsed ? '0' : '200px'};
  overflow: hidden;
  transition: max-height 0.3s ease;
  overflow-y: auto;
  opacity: ${props => props.isCollapsed ? 0 : 1};
  height: ${props => props.isCollapsed ? '0' : 'auto'};
`;

const ActionBar = styled.div`
  display: flex;
  justify-content: space-between;
  padding: 8px 12px;
  background: #f8f9fa;
  border-top: 1px solid #e9ecef;
`;

const ActionButton = styled.button`
  background: none;
  border: 1px solid #ced4da;
  border-radius: 4px;
  padding: 4px 8px;
  font-size: 12px;
  color: #6c757d;
  cursor: pointer;
  
  &:hover {
    background: #e9ecef;
    color: #343a40;
  }
`;

// Analytics section styles
const AnalyticsSection = styled.div`
  padding: 8px 12px;
  background: #f8f9fa;
  border-top: 1px solid #e9ecef;
  font-size: 12px;
`;

const AnalyticsTitle = styled.div`
  display: flex;
  justify-content: space-between;
  font-weight: 600;
  margin-bottom: 8px;
  color: #495057;
`;

const AnalyticsGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
`;

const AnalyticsItem = styled.div`
  background: white;
  border: 1px solid #e9ecef;
  border-radius: 4px;
  padding: 6px 8px;
`;

const DepthSelector = styled.div`
  display: flex;
  align-items: center;
  
  input {
    width: 60px;
    padding: 2px 4px;
    border: 1px solid #ced4da;
    border-radius: 3px;
    font-size: 11px;
  }
`;

// Depth analysis table styles
const DepthTable = styled.table`
  width: 100%;
  border-collapse: collapse;
  margin-top: 8px;
  font-size: 11px;
  
  th, td {
    padding: 5px 8px;
    text-align: right;
    border: 1px solid #dee2e6;
  }
  
  th {
    background-color: #f8f9fa;
    color: #495057;
    font-weight: 600;
  }
  
  tr:nth-child(even) {
    background-color: #f8f9fa;
  }
  
  td.depth-cell {
    text-align: center;
    font-weight: 600;
  }
  
  .progress-bar-cell {
    width: 25%;
    position: relative;
    padding: 0;
    
    .progress-container {
      height: 100%;
      width: 100%;
      padding: 6px 0;
    }
    
    .progress-bar {
      height: 8px;
      background-color: #007bff;
      border-radius: 2px;
    }
  }
`;

const AnalyticsTabs = styled.div`
  display: flex;
  margin-bottom: 8px;
  border-bottom: 1px solid #dee2e6;
`;

const AnalyticsTab = styled.div<{ isActive: boolean }>`
  padding: 6px 12px;
  font-size: 11px;
  cursor: pointer;
  border-bottom: 2px solid ${props => props.isActive ? '#007bff' : 'transparent'};
  color: ${props => props.isActive ? '#007bff' : '#6c757d'};
  
  &:hover {
    color: ${props => props.isActive ? '#007bff' : '#495057'};
  }
`;

// Get maximum tree depth
function getTreeMaxDepth(node: VisualNode, currentDepth: number = 0): number {
  if (!node || !node.children || node.children.length === 0) {
    return currentDepth;
  }
  
  return Math.max(...node.children.map(child => 
    getTreeMaxDepth(child, currentDepth + 1)
  ));
}

// Count total nodes in tree
function countNodes(node: VisualNode): number {
  if (!node) return 0;
  
  let count = 1; // Current node
  
  if (node.children && node.children.length > 0) {
    for (const child of node.children) {
      count += countNodes(child);
    }
  }
  
  return count;
}

interface DepthCoverageProps {
  coverageByDepth: number[];
  nodeCounts: number[];
}

// Depth-wise coverage table component
const DepthCoverageTable: React.FC<DepthCoverageProps> = ({
  coverageByDepth,
  nodeCounts
}) => {
  return (
    <DepthTable>
      <thead>
        <tr>
          <th>Depth</th>
          <th>Coverage</th>
          <th>Nodes</th>
          <th>Visual</th>
        </tr>
      </thead>
      <tbody>
        {coverageByDepth.map((coverage, index) => (
          <tr key={index}>
            <td className="depth-cell">{index + 1}</td>
            <td>{coverage.toFixed(2)}%</td>
            <td>{nodeCounts[index]}</td>
            <td className="progress-bar-cell">
              <div className="progress-container">
                <div 
                  className="progress-bar" 
                  style={{ width: `${Math.min(100, coverage)}%` }}
                />
              </div>
            </td>
          </tr>
        ))}
      </tbody>
    </DepthTable>
  );
};

interface DebugPanelProps {
  connected: boolean;
  messages: Array<{
    direction: 'sent' | 'received';
    timestamp: number;
    data: any;
  }>;
  onClose: () => void;
  visualTree?: VisualNode | null;
}

export const DebugPanel: React.FC<DebugPanelProps> = ({
  connected,
  messages,
  onClose,
  visualTree
}) => {
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [collapsedMessages, setCollapsedMessages] = useState<Record<number, boolean>>({});
  const [coverageDepth, setCoverageDepth] = useState<number>(10);
  const [probabilityCoverage, setProbabilityCoverage] = useState<number>(0);
  const [depthCoverage, setDepthCoverage] = useState<{
    coverageByDepth: number[];
    nodeCounts: number[];
  }>({
    coverageByDepth: [],
    nodeCounts: []
  });
  const [treeMaxDepth, setTreeMaxDepth] = useState<number>(0);
  const [activeTab, setActiveTab] = useState<'summary' | 'depth'>('summary');
  

  // Auto-scroll when new messages arrive
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
    
    // Set new messages to expanded state by default
    const newMessage = messages.length > 0 ? messages[messages.length - 1] : null;
    if (newMessage) {
      setCollapsedMessages(prev => ({
        ...prev,
        [messages.length - 1]: false
      }));
    }
  }, [messages.length]);

  // Format message timestamp
  const formatTime = (timestamp: number) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  };

  // Toggle message collapse/expand
  const toggleMessage = (index: number) => {
    setCollapsedMessages(prev => ({
      ...prev,
      [index]: !prev[index]
    }));
  };

  // Copy functionality
  const copyMessagesToClipboard = () => {
    const text = messages.map(msg => {
      const time = formatTime(msg.timestamp);
      const direction = msg.direction === 'sent' ? 'SENT' : 'RECEIVED';
      return `[${time}] ${direction}:\n${JSON.stringify(msg.data, null, 2)}`;
    }).join('\n\n');
    
    navigator.clipboard.writeText(text).then(() => {
      alert('Debug messages copied to clipboard');
    });
  };

  return (
    <DebugContainer>
      <DebugHeader>
        <HeaderTitle>Debug Console</HeaderTitle>
        <CloseButton onClick={onClose}>×</CloseButton>
      </DebugHeader>
      
      <StatusBar connected={connected}>
        <span>Status: {connected ? 'Connected' : 'Disconnected'}</span>
        <span>Messages: {messages.length}</span>
      </StatusBar>
      
      {/* Analytics section */}
      <AnalyticsSection>
        <AnalyticsTitle>
          <span>Tree Analytics</span>
          <span>{visualTree ? '✓ Available' : '✗ Not Available'}</span>
        </AnalyticsTitle>
        
        <AnalyticsTabs>
          <AnalyticsTab 
            isActive={activeTab === 'summary'} 
            onClick={() => setActiveTab('summary')}
          >
            Summary
          </AnalyticsTab>
          <AnalyticsTab 
            isActive={activeTab === 'depth'} 
            onClick={() => setActiveTab('depth')}
          >
            Depth Analysis
          </AnalyticsTab>
        </AnalyticsTabs>
        
        {activeTab === 'summary' ? (
          <>
            <AnalyticsGrid>
              <AnalyticsItem>
                <div style={{ fontSize: '11px', color: '#6c757d' }}>Probability Coverage:</div>
                <div style={{ fontWeight: 'bold', fontSize: '14px' }}>
                  {probabilityCoverage.toFixed(2)}%
                </div>
              </AnalyticsItem>
              <AnalyticsItem>
                <div style={{ fontSize: '11px', color: '#6c757d' }}>Node Count:</div>
                <div style={{ fontWeight: 'bold', fontSize: '14px' }}>
                  {visualTree ? countNodes(visualTree) : 0}
                </div>
              </AnalyticsItem>
            </AnalyticsGrid>
            
            <AnalyticsGrid style={{ marginTop: '8px' }}>
              <AnalyticsItem>
                <div style={{ fontSize: '11px', color: '#6c757d' }}>Current Tree Depth:</div>
                <div style={{ fontWeight: 'bold', fontSize: '14px' }}>
                  {treeMaxDepth}
                </div>
              </AnalyticsItem>
              <AnalyticsItem>
                <div style={{ fontSize: '11px', color: '#6c757d' }}>Analysis Depth:</div>
                <DepthSelector>
                  <input 
                    id="coverageDepth"
                    type="number" 
                    min="1" 
                    max="20" 
                    value={coverageDepth}
                    onChange={(e) => setCoverageDepth(Number(e.target.value))} 
                  />
                </DepthSelector>
              </AnalyticsItem>
            </AnalyticsGrid>
          </>
        ) : (
          <DepthCoverageTable 
            coverageByDepth={depthCoverage.coverageByDepth} 
            nodeCounts={depthCoverage.nodeCounts}
          />
        )}
      </AnalyticsSection>
      
      <MessagesContainer>
        {messages.map((msg, idx) => {
          const isCollapsed = !!collapsedMessages[idx];
          
          return (
            <Message key={idx} direction={msg.direction} isCollapsed={isCollapsed}>
              <MessageHeader>
                <div style={{ display: 'flex', alignItems: 'center' }}>
                  <MessageDirection direction={msg.direction}>
                    {msg.direction === 'sent' ? 'SENT' : 'RECEIVED'}
                  </MessageDirection>
                  <span 
                    className="message-toggle" 
                    onClick={() => toggleMessage(idx)}
                    title={isCollapsed ? "expand" : "collapse"}
                  >
                    {isCollapsed ? '▶' : '▼'}
                  </span>
                </div>
                <span>{formatTime(msg.timestamp)}</span>
              </MessageHeader>
              <MessageContent isCollapsed={isCollapsed}>
                {JSON.stringify(msg.data, null, 2)}
              </MessageContent>
            </Message>
          );
        })}
        <div ref={messagesEndRef} />
      </MessagesContainer>
      
      <ActionBar>
        <div>
          <ActionButton onClick={() => {
            // Toggle all messages collapse/expand
            const allCollapsed = Object.values(collapsedMessages).every(val => val);
            const newState = !allCollapsed;
            
            const newCollapsedState = messages.reduce((acc, _, idx) => {
              acc[idx] = newState;
              return acc;
            }, {} as Record<number, boolean>);
            
            setCollapsedMessages(newCollapsedState);
          }}>
            {Object.values(collapsedMessages).every(val => val) ? "expand all" : "collapse all"}
          </ActionButton>
        </div>
        <div>
          <ActionButton onClick={copyMessagesToClipboard}>
            Copy to Clipboard
          </ActionButton>
        </div>
      </ActionBar>
    </DebugContainer>
  );
};