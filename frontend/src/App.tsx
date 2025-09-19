import React, { useState, useCallback, useRef, useEffect } from 'react';
import { ParameterPanel } from './components/parameterPanel';
import { TokenTreeVisualizer } from './components/tokenTreePanel';
import { TokenStreamPanel } from './components/tokenStreamPanel';
import { EvaluatedPathsPanel } from './components/evaluatedPathsPanel';
import { useWebSocket } from './hooks/useWebSocket';
import { useTreeManager } from './hooks/useTreeManager';
import { generateSequenceFromNode } from './utils/treeTransform';
import { CompletedSequence } from './types/types';
import styled from 'styled-components';

const AppContainer = styled.div`
  display: grid;
  grid-template-rows: 0.5fr 1fr;
  height: 100vh;
  gap: 16px;
  padding: 16px;
  background: #f8f9fa;
  box-sizing: border-box;
  max-width: 1920px;
  margin: 0 auto;
`;

const TopSection = styled.div`
  min-height: 0;
`;

const BottomSection = styled.div`
  display: grid;
  grid-template-columns: 30% 70%;
  gap: 16px;
  min-height: 0;
  overflow: hidden;
`;

const TreeVisualizerContainer = styled.div`
  border: 1px solid #e9ecef;
  border-radius: 8px;
  padding: 16px;
  background: white;
  overflow: auto;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
  height: 100%;
`;

function App() {
  const [tempSequence, setTempSequence] = useState<CompletedSequence | null>(null);
  
  
  const { 
    state: webSocketState, 
    sendMessage, 
    connect, 
    disconnect, 
    clearMessages 
  } = useWebSocket();
  
  const {
    visualTree,
    completedSequences,
    selectedToken,
    alternativeTokens,
    modelStatus,
    loadingProgress,
    statusMessage,
    isGenerating,
    evaluatedNodeIds,
    evaluatedPathsTotal,
    evaluationFilters,
    resetAll,
    handleTokenSelect,
    handleSequenceSelect: originalHandleSequenceSelect,
    handleNodePin,
    handleNodeEvaluate,
    toggleEvaluationFilter,
    handleRequestTokenGeneration,
    getSelectedSequence,
    isNodeGenerating, 
    tokenGenerationRequestRef,
    updateParameters
  } = useTreeManager({
    webSocketState,
    sendMessage,
    clearMessages,
    tempSequence
  });

  const handleSequenceSelect = useCallback((sequence: CompletedSequence) => {
    setTempSequence(null); 
    originalHandleSequenceSelect(sequence);
  }, [originalHandleSequenceSelect]);

 
  const handleTreeNodeSelect = useCallback((sequenceId: string, tokenIndex: number) => {
    if (sequenceId === '' && tokenIndex === -1) {
      handleTokenSelect('', -1);
      setTempSequence(null);
      return;
    }
    
    // generate temp sequence if a tree node is selected
    if (sequenceId.startsWith('temp-')) {
      const nodeId = sequenceId.replace('temp-', '');
      
      if (visualTree) {
        const generatedSequence = generateSequenceFromNode(visualTree, nodeId);
        if (generatedSequence) {
          setTempSequence(generatedSequence);
          // find the actual index of the clicked node in the generated sequence
          const actualTokenIndex = generatedSequence.tokens.findIndex(t => t.nodeId === nodeId);
          // set selectedToken to point to the temp sequence
          handleTokenSelect(generatedSequence.id, actualTokenIndex >= 0 ? actualTokenIndex : 0);
          return;
        }
      }
    }
    
    // selected a normal completed sequence
    setTempSequence(null);
    handleTokenSelect(sequenceId, tokenIndex);
  }, [visualTree, handleTokenSelect]);


  return (
    <AppContainer>
      <TopSection>
        <ParameterPanel
          connected={webSocketState.connected}
          error={webSocketState.error}
          messages={webSocketState.messages}
          onSendMessage={sendMessage}
          onConnect={connect}
          onDisconnect={disconnect}
          onClearMessages={resetAll}
          onRequestTokenGeneration={tokenGenerationRequestRef}
          modelStatus={modelStatus}
          loadingProgress={loadingProgress}
          statusMessage={statusMessage}
          isGenerating={isGenerating}
          hasTree={!!visualTree}
          onUpdateParameters={updateParameters}
          completedSequence={tempSequence || getSelectedSequence()}
          selectedToken={selectedToken}
          onTokenSelect={handleTokenSelect}
          alternatives={alternativeTokens.map(alt => ({ 
            token: alt.token, 
            probability: alt.probability,
            nodeId: alt.nodeId
          }))}
          completedSequences={completedSequences}
          onSequenceSelect={handleSequenceSelect}
        />
      </TopSection>
      <BottomSection>
        <EvaluatedPathsPanel
          visualTree={visualTree}
          evaluatedNodeIds={evaluatedNodeIds}
          onPathSelect={(nodeId) => {
            handleTreeNodeSelect(`temp-${nodeId}`, 0);
          }}
          filters={evaluationFilters}
          onToggleFilter={toggleEvaluationFilter}
        />
        <TreeVisualizerContainer>
          {visualTree ? (
            <TokenTreeVisualizer 
              node={visualTree} 
              selectedToken={selectedToken}
              completedSequences={completedSequences}
              tempSequence={tempSequence} 
              onNodePin={handleNodePin}
              onNodeEvaluate={handleNodeEvaluate}
              onGenerateFromNode={handleRequestTokenGeneration}
              isNodeGenerating={isNodeGenerating}
              onTokenSelect={handleTreeNodeSelect}
              evaluatedNodeIds={evaluatedNodeIds}
              evaluatedPathsTotal={evaluatedPathsTotal}
              evaluationFilters={evaluationFilters}
            />
          ) : (
            <div style={{ 
              textAlign: 'center', 
              color: '#6c757d', 
              padding: '40px',
              fontSize: '14px'
            }}>
              No tree data yet. Start inference to see visualization.
            </div>
          )}
        </TreeVisualizerContainer>
      </BottomSection>
    </AppContainer>
  );
}

export default App;