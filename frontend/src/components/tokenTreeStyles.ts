import styled from 'styled-components';

// Slider text
export const SliderText = styled.div`
  font-size: 13px;
  font-weight: 500;
  color: #333;
  margin: 4px 0;
  text-align: center;
  width: 100%;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
`;

export const SliderLabel = styled.div`
  font-size: 10px;
  color: #888;
  text-align: center;
  width: 100%;
  margin: 2px 0;
`;

// Main container
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

// Tooltip styles
export const Tooltip = styled.div`
  position: absolute;
  padding: 10px;
  background: white;
  border: 1px solid #dee2e6;
  border-radius: 6px;
  pointer-events: none;
  font-size: 13px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
  opacity: 0;
  z-index: 1000;

  .token-info {
    display: flex;
    align-items: center;
    padding: 4px 8px;
    margin: 2px 0;
    background: #f8f9fa;
    border-radius: 4px;
  }

  .token-prob {
    margin-left: 8px;
    color: #6c757d;
    font-size: 12px;
  }
`;

// Zoom control
export const ZoomControl = styled.div`
  position: absolute;
  right: 16px;
  top: 84px;
  background: rgba(255, 255, 255, 0.98);
  padding: 8px;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.12);
  border: 1px solid #e0e0e0;
  z-index: 1000;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  width: 56px;
`;

// Home button
export const HomeButton = styled.button`
  width: 40px;
  height: 32px;
  background-color: white;
  color: #333;
  border: 1px solid #d0d0d0;
  border-radius: 4px;
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

// Vertical slider container
export const VerticalSliderContainer = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  height: 280px;
  width: 100%;
  padding: 0;
`;

// Slider button
export const SliderButton = styled.button`
  width: 40px;
  height: 32px;
  background-color: white;
  color: #333;
  border: 1px solid #d0d0d0;
  border-radius: 4px;
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

// Vertical slider container with ticks
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

// Vertical slider
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

// Slider ticks container (disabled)
export const SliderTicks = styled.div`
  display: none;
`;

// Individual tick (disabled)
export const SliderTick = styled.div`
  display: none;
`;

// Tick label
export const SliderTickLabel = styled.span`
  font-size: 8px;
  color: #666;
  white-space: nowrap;
`;

// Merged node container
export const MergedNodeContainer = styled.g`
  .merged-node-bg {
    fill: white;
    stroke: #dee2e6;
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

// Token text
export const TokenText = styled.text`
  font-family: monospace;
  font-size: 12px;
  user-select: none;
`;

// Token separator
export const TokenSeparator = styled.rect`
  fill: #e9ecef;
  rx: 2;
`;

// Node context menu styles
export const ContextMenu = styled.div`
  position: absolute;
  background: white;
  border: 1px solid #dee2e6;
  border-radius: 6px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.12);
  z-index: 1500;
  min-width: 180px;
  overflow: hidden;
`;

// Context menu item
export const ContextMenuItem = styled.div`
  padding: 10px 16px;
  cursor: pointer;
  font-size: 14px;
  
  &:hover {
    background-color: #f8f9fa;
  }
`;

// Context menu divider
export const ContextMenuDivider = styled.div`
  height: 1px;
  background-color: #e9ecef;
  margin: 4px 0;
`;

// Node state icon container
export const NodeStateIcon = styled.g`
  cursor: pointer;
`;

// Pinned node style
export const PinnedNodeIndicator = styled.circle`
  fill: #ff9800;
  stroke: #fff;
  stroke-width: 1;
`;

// Toggle control container
export const ToggleControl = styled.div`
  position: absolute;
  right: 16px;
  top: 16px;
  background: rgba(255, 255, 255, 0.98);
  padding: 8px;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.12);
  border: 1px solid #e0e0e0;
  z-index: 1000;
  display: flex;
  gap: 8px;
`;

// Alignment toggle button
export const AlignmentButton = styled.button<{isActive: boolean}>`
  width: 36px;
  height: 36px;
  background-color: ${props => props.isActive ? '#007bff' : '#6c757d'};
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  font-weight: bold;
  transition: all 0.2s ease;
  
  &:hover {
    background-color: ${props => props.isActive ? '#0056b3' : '#5a6268'};
    transform: scale(1.05);
  }
`;

// Sankey link toggle button
export const SankeyButton = styled.button<{isActive: boolean}>`
  width: 36px;
  height: 36px;
  background-color: ${props => props.isActive ? '#17a2b8' : '#6c757d'};
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  font-weight: bold;
  transition: all 0.2s ease;
  
  &:hover {
    background-color: ${props => props.isActive ? '#138496' : '#5a6268'};
    transform: scale(1.05);
  }
`;

export const MarkNotification = styled.div<{ x: number; y: number; category: 'good' | 'bad' }>`
  position: absolute;
  left: ${props => props.x}px;
  top: ${props => props.y - 30}px;
  background-color: ${props => props.category === 'good' ? '#28a745' : '#dc3545'};
  color: white;
  padding: 4px 8px;
  border-radius: 12px;
  font-size: 11px;
  font-weight: bold;
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
    border-top: 4px solid ${props => props.category === 'good' ? '#28a745' : '#dc3545'};
  }
`;