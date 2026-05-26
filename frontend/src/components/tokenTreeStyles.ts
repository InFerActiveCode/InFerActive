import styled from 'styled-components';

// 슬라이더 텍스트
export const SliderText = styled.div`
  font-size: 13px;
  font-weight: 500;
  color: var(--genesis-text-primary);
  margin: 4px 0;
  text-align: center;
  width: 100%;
`;

export const SliderLabel = styled.div`
  font-size: 10px;
  color: var(--genesis-text-secondary);
  text-align: center;
  width: 100%;
  margin: 2px 0;
`;

// 메인 컨테이너
export const TreeContainer = styled.div`
  width: 100%;
  height: 100%;
  overflow: hidden;
  cursor: grab;
  position: relative;
  
  &:active {
    cursor: grabbing;
  }
`;

// 툴팁 스타일
export const Tooltip = styled.div`
  position: absolute;
  padding: 10px;
  background: var(--genesis-surface);
  border: 1px solid var(--genesis-border);
  border-radius: var(--genesis-radius-sm);
  pointer-events: none;
  font-size: 13px;
  color: var(--genesis-text-primary);
  box-shadow: 0 2px 8px rgba(0,0,0,0.12);
  opacity: 0;
  z-index: 1000;

  .token-info {
    display: flex;
    align-items: center;
    padding: 4px 8px;
    margin: 2px 0;
    background: #f8f9fa;
    border-radius: var(--genesis-radius-xs);
  }

  .token-prob {
    margin-left: 8px;
    color: var(--genesis-text-secondary);
    font-size: 12px;
  }
`;

// 홈 버튼 컨트롤 (왼쪽 상단)
export const HomeControl = styled.div`
  position: absolute;
  left: 8px;
  top: 8px;
  background: rgba(255, 255, 255, 0.98);
  padding: 8px;
  border-radius: var(--genesis-radius-md);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.12);
  border: 1px solid var(--genesis-border);
  z-index: 1000;
  display: flex;
  align-items: center;
  justify-content: center;
`;

// 슬라이더 컨트롤 (오른쪽 상단)
export const ZoomControl = styled.div`
  position: absolute;
  right: 16px;
  top: 50px;
  background: rgba(255, 255, 255, 0.98);
  padding: 8px;
  border-radius: 8px;
  box-shadow: 0 6px 18px rgba(15, 23, 42, 0.1);
  border: 1px solid var(--genesis-border);
  z-index: 1000;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  width: 56px;
`;

// 홈 버튼
export const HomeButton = styled.button`
  width: 40px;
  height: 32px;
  background-color: var(--genesis-surface);
  color: var(--genesis-text-primary);
  border: 1px solid var(--genesis-border-input);
  border-radius: var(--genesis-radius-xs);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.15s;
  
  &:hover {
    background-color: #f5f5f5;
    border-color: #999;
  }
  
  &:active {
    background-color: #e0e0e0;
    transform: scale(0.98);
  }
`;

// 수직 슬라이더 컨테이너
export const VerticalSliderContainer = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  height: 280px;
  width: 100%;
  padding: 0;
`;

// 슬라이더 버튼
export const SliderButton = styled.button`
  width: 40px;
  height: 32px;
  background-color: var(--genesis-surface);
  color: var(--genesis-text-primary);
  border: 1px solid var(--genesis-border-input);
  border-radius: var(--genesis-radius-xs);
  cursor: pointer;
  font-size: 18px;
  font-weight: 500;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.15s;
  
  &:hover {
    background-color: #f5f5f5;
    border-color: #999;
  }
  
  &:active {
    background-color: #e0e0e0;
    transform: scale(0.98);
  }
`;

// 수직 슬라이더 컨테이너 (슬라이더와 눈금을 함께 포함)
export const SliderWithTicks = styled.div`
  position: relative;
  width: 40px;
  height: 180px;
  margin: 4px 0;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
`;

// 수직 슬라이더
export const VerticalSlider = styled.input`
  -webkit-appearance: none;
  -moz-appearance: none;
  appearance: none;
  width: 180px;
  height: 6px;
  background: #e8e8e8;
  outline: none;
  border-radius: 3px;
  cursor: pointer;
  transform: rotate(-90deg);
  transform-origin: center;
  
  &::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 24px;
    height: 32px;
    border-radius: 3px;
    background: white;
    cursor: pointer;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.3);
    border: 1px solid #999;
    transition: all 0.15s;
  }
  
  &::-webkit-slider-thumb:hover {
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.4);
    background: #f8f8f8;
    transform: scale(1.1);
  }
  
  &::-moz-range-thumb {
    width: 32px;
    height: 32px;
    border-radius: 3px;
    background: white;
    cursor: pointer;
    border: 1px solid #999;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.3);
    transition: all 0.15s;
  }
  
  &::-moz-range-thumb:hover {
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.4);
    background: #f8f8f8;
    transform: scale(1.1);
  }
  
  &::-moz-range-track {
    width: 180px;
    height: 6px;
    background: #e8e8e8;
    border-radius: 3px;
  }
`;

// 슬라이더 눈금 컨테이너 (비활성화)
export const SliderTicks = styled.div`
  display: none;
`;

// 개별 눈금 (비활성화)
export const SliderTick = styled.div`
  display: none;
`;

// 눈금 라벨
export const SliderTickLabel = styled.span`
  font-size: 8px;
  color: var(--genesis-text-secondary);
  white-space: nowrap;
`;

// 병합된 노드 컨테이너
export const MergedNodeContainer = styled.g`
  .merged-node-bg {
    fill: white;
    stroke: #e9ecef;
    stroke-width: 1;
    rx: 6;
  }

  .merged-node-text {
    font-family: monospace;
    font-size: 12px;
  }

  .token-separator {
    fill: #e9ecef;
    rx: 2;
  }

  &:hover {
    .merged-node-bg {
      stroke: #adb5bd;
    }
  }
`;

// 토큰 텍스트
export const TokenText = styled.text`
  font-family: monospace;
  font-size: 12px;
  user-select: none;
`;

// 토큰 구분자
export const TokenSeparator = styled.rect`
  fill: #e9ecef;
  rx: 2;
`;

// 노드 컨텍스트 메뉴 스타일
export const ContextMenu = styled.div`
  position: absolute;
  background: var(--genesis-surface);
  border: 1px solid var(--genesis-border);
  border-radius: var(--genesis-radius-sm);
  box-shadow: 0 2px 12px rgba(0,0,0,0.12);
  z-index: 1500;
  min-width: 180px;
  overflow: hidden;
`;

// 컨텍스트 메뉴 항목
export const ContextMenuItem = styled.div`
  padding: 10px 16px;
  cursor: pointer;
  font-size: 13px;
  color: var(--genesis-text-primary);
  transition: background-color 0.1s;

  &:hover {
    background-color: var(--genesis-background);
  }
`;

// 컨텍스트 메뉴 구분선
export const ContextMenuDivider = styled.div`
  height: 1px;
  background-color: var(--genesis-border);
  margin: 4px 0;
`;

// 노드 상태 표시용 아이콘 컨테이너
export const NodeStateIcon = styled.g`
  cursor: pointer;
`;

// 고정된 노드 스타일
export const PinnedNodeIndicator = styled.circle`
  fill: #ff9800;
  stroke: #fff;
  stroke-width: 1;
`;

// 토글 컨트롤 컨테이너
export const ToggleControl = styled.div`
  position: absolute;
  right: 16px;
  bottom: 16px;
  background: rgba(255, 255, 255, 0.98);
  padding: 8px;
  border-radius: var(--genesis-radius-md);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.12);
  border: 1px solid var(--genesis-border);
  z-index: 1000;
  display: flex;
  gap: 8px;
`;

// 정렬 토글 버튼
export const AlignmentButton = styled.button<{isActive: boolean}>`
  width: 36px;
  height: 36px;
  background-color: ${props => props.isActive ? 'var(--genesis-primary)' : '#6c757d'};
  color: white;
  border: none;
  border-radius: var(--genesis-radius-sm);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  font-weight: bold;
  transition: background-color 0.15s;

  &:hover {
    background-color: ${props => props.isActive ? 'var(--genesis-primary-hover)' : '#5a6268'};
  }
`;

// Sankey 링크 토글 버튼
export const SankeyButton = styled.button<{isActive: boolean}>`
  width: 36px;
  height: 36px;
  background-color: ${props => props.isActive ? '#17a2b8' : '#6c757d'};
  color: white;
  border: none;
  border-radius: var(--genesis-radius-sm);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  font-weight: bold;
  transition: background-color 0.15s;

  &:hover {
    background-color: ${props => props.isActive ? '#138496' : '#5a6268'};
  }
`;

export const MarkNotification = styled.div<{ x: number; y: number; category: 'good' | 'bad' }>`
  position: absolute;
  left: ${props => props.x}px;
  top: ${props => props.y - 30}px;
  background-color: ${props => props.category === 'good' ? 'var(--genesis-success)' : 'var(--genesis-error)'};
  color: white;
  padding: 4px 8px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 700;
  white-space: nowrap;
  pointer-events: none;
  z-index: 10000;
  animation: fadeInOut 1s ease-out forwards;
  
  @keyframes fadeInOut {
    0% {
      opacity: 0;
      transform: translateY(5px);
    }
    20% {
      opacity: 1;
      transform: translateY(0);
    }
    80% {
      opacity: 1;
      transform: translateY(0);
    }
    100% {
      opacity: 0;
      transform: translateY(-5px);
    }
  }
  
  &::after {
    content: '';
    position: absolute;
    bottom: -4px;
    left: 50%;
    transform: translateX(-50%);
    width: 0;
    height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 4px solid ${props => props.category === 'good' ? 'var(--genesis-success)' : 'var(--genesis-error)'};
  }
`;

export const EvalActionBar = styled.div<{ x: number; y: number }>`
  position: absolute;
  left: ${props => props.x}px;
  top: ${props => props.y - 38}px;
  transform: translateX(-50%);
  display: flex;
  gap: 2px;
  background: var(--genesis-surface);
  border: 1px solid var(--genesis-border);
  border-radius: var(--genesis-radius-md);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
  padding: 3px;
  z-index: 1200;
  animation: evalBarFadeIn 0.15s ease-out;

  @keyframes evalBarFadeIn {
    from { opacity: 0; transform: translateX(-50%) translateY(4px); }
    to { opacity: 1; transform: translateX(-50%) translateY(0); }
  }

  &::after {
    content: '';
    position: absolute;
    bottom: -6px;
    left: 50%;
    transform: translateX(-50%);
    width: 0;
    height: 0;
    border-left: 6px solid transparent;
    border-right: 6px solid transparent;
    border-top: 6px solid white;
    filter: drop-shadow(0 1px 1px rgba(0, 0, 0, 0.1));
  }
`;

export const EvalActionButton = styled.button<{ variant: 'good' | 'bad' | 'root'; isActive?: boolean }>`
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 4px 10px;
  border: none;
  border-radius: 6px;
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.1s, color 0.1s;
  white-space: nowrap;

  background: ${props => {
    if (!props.isActive) return 'transparent';
    if (props.variant === 'good') return '#e8f5e9';
    if (props.variant === 'bad') return '#ffebee';
    return '#e3f2fd';
  }};
  color: ${props => {
    if (!props.isActive) return '#555';
    if (props.variant === 'good') return '#2e7d32';
    if (props.variant === 'bad') return '#c62828';
    return '#1565c0';
  }};

  &:hover {
    background: ${props => {
      if (props.variant === 'good') return '#e8f5e9';
      if (props.variant === 'bad') return '#ffebee';
      return '#e3f2fd';
    }};
    color: ${props => {
      if (props.variant === 'good') return '#2e7d32';
      if (props.variant === 'bad') return '#c62828';
      return '#1565c0';
    }};
  }

  &:disabled {
    opacity: 0.4;
    cursor: not-allowed;
    &:hover {
      background: transparent;
      color: #999;
    }
  }
`;
