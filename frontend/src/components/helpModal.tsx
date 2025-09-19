import React, { useEffect } from 'react';
import styled from 'styled-components';

const ModalOverlay = styled.div`
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
`;

const ModalContent = styled.div`
  background: white;
  border-radius: 16px;
  padding: 40px;
  max-width: 1200px;
  width: 90%;
  max-height: 90vh;
  overflow-y: auto;
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
  display: flex;
  flex-direction: column;
`;

const ModalHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 20px;
  padding-bottom: 16px;
  border-bottom: 1px solid #e9ecef;
`;

const Title = styled.h2`
  margin: 0;
  color: #343a40;
  font-size: 26px;
  font-weight: 700;
  display: flex;
  align-items: center;
  gap: 12px;
`;

const CloseButton = styled.button`
  background: none;
  border: none;
  font-size: 24px;
  color: #6c757d;
  cursor: pointer;
  padding: 4px;
  display: flex;
  align-items: center;
  justify-content: center;
  
  &:hover {
    color: #343a40;
  }
`;

const ContentGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr 1.2fr;
  gap: 0;
  flex: 1;
  overflow-y: auto;
  
  > div:first-child {
    padding-right: 40px;
  }
  
  > div:last-child {
    padding-left: 40px;
    border-left: 1px solid #e9ecef;
  }
  
  @media (max-width: 968px) {
    grid-template-columns: 1fr;
    
    > div:first-child {
      padding-right: 0;
      padding-bottom: 32px;
    }
    
    > div:last-child {
      padding-left: 0;
      border-left: none;
      padding-top: 32px;
      border-top: 1px solid #e9ecef;
    }
  }
`;

const Section = styled.div`
  margin-bottom: 32px;
  
  &:last-child {
    margin-bottom: 0;
  }
`;

const SectionTitle = styled.h3`
  color: #343a40;
  font-size: 20px;
  font-weight: 600;
  margin: 0 0 16px 0;
  display: flex;
  align-items: center;
  gap: 8px;
`;

const VideoContainer = styled.div`
  width: 100%;
  aspect-ratio: 16 / 9;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  background: #000;
`;

const VideoElement = styled.video`
  width: 100%;
  height: 100%;
  object-fit: contain;
`;

const Description = styled.div`
  color: #495057;
  font-size: 16px;
  line-height: 1.8;
  
  p {
    margin: 0 0 16px 0;
    
    &:last-child {
      margin-bottom: 0;
    }
  }
  
  strong {
    color: #343a40;
  }
`;

const ScreenshotContainer = styled.div`
  background: #f8f9fa;
  border-radius: 12px;
  padding: 20px;
  border: 1px solid #e9ecef;
  margin-bottom: 20px;
`;

const ScreenshotImage = styled.img`
  width: 100%;
  height: auto;
  border-radius: 8px;
  border: 1px solid #dee2e6;
  display: block;
`;

const AnnotationList = styled.ol`
  margin: 0;
  padding: 0;
  list-style: none;
  color: #495057;
  font-size: 14px;
  line-height: 1.8;
  counter-reset: item;
  
  li {
    margin-bottom: 12px;
    padding-left: 36px;
    position: relative;
    counter-increment: item;
    
    &:last-child {
      margin-bottom: 0;
    }
    
    &::before {
      content: counter(item);
      position: absolute;
      left: 0;
      top: 0;
      width: 28px;
      height: 28px;
      background-color: #0066cc;
      color: white;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 16px;
      font-weight: bold;
    }
  }
  
  strong {
    color: #343a40;
    font-weight: 600;
  }
`;

const SimpleList = styled.ol`
  margin: 0;
  padding-left: 20px;
  color: #495057;
  font-size: 14px;
  line-height: 1.8;
  list-style-type: decimal;
  
  li {
    margin-bottom: 8px;
    
    &:last-child {
      margin-bottom: 0;
    }
  }
  
  strong {
    color: #343a40;
    font-weight: 600;
  }
`;

const ContactInfo = styled.div`
  background: #fff5f5;
  border: 1px solid #fee;
  border-radius: 8px;
  padding: 20px;
  margin-top: 16px;
`;

const ContactText = styled.p`
  color: #495057;
  font-size: 14px;
  line-height: 1.6;
  margin: 0;
  
  a {
    color: #007bff;
    text-decoration: none;
    font-weight: 500;
    
    &:hover {
      text-decoration: underline;
    }
  }
`;

const ButtonGroup = styled.div`
  display: flex;
  gap: 12px;
  align-items: center;
  margin-top: 20px;
  padding-top: 16px;
  border-top: 1px solid #e9ecef;
  justify-content: center;
`;

const Button = styled.button<{ variant?: 'primary' | 'secondary' }>`
  padding: 12px 24px;
  border-radius: 6px;
  border: none;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
  
  ${props => props.variant === 'primary' ? `
    background-color: #007bff;
    color: white;
    
    &:hover {
      background-color: #0056b3;
    }
  ` : `
    background-color: #6c757d;
    color: white;
    
    &:hover {
      background-color: #5a6268;
    }
  `}
`;

const CheckboxContainer = styled.label`
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 14px;
  color: #6c757d;
  cursor: pointer;
  
  input[type="checkbox"] {
    margin: 0;
  }
`;

interface HelpModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const HelpModal: React.FC<HelpModalProps> = ({ isOpen, onClose }) => {
  const [dontShowAgain, setDontShowAgain] = React.useState(false);

  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };

    if (isOpen) {
      document.addEventListener('keydown', handleEscape);
      document.body.style.overflow = 'hidden';
    }

    return () => {
      document.removeEventListener('keydown', handleEscape);
      document.body.style.overflow = 'unset';
    };
  }, [isOpen, onClose]);

  const handleClose = () => {
    if (dontShowAgain) {
      localStorage.setItem('inferactive_help_dismissed', 'true');
    }
    onClose();
  };

  const handleStartDemo = () => {
    if (dontShowAgain) {
      localStorage.setItem('inferactive_help_dismissed', 'true');
    }
    onClose();
  };

  const handleOverlayClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) {
      handleClose();
    }
  };

  if (!isOpen) return null;

  return (
    <ModalOverlay onClick={handleOverlayClick}>
      <ModalContent>
        <ModalHeader>
          <Title>
            InFerActive: Interactive Inference of LLM
          </Title>
          <CloseButton onClick={handleClose}>×</CloseButton>
        </ModalHeader>
        
        <ContentGrid>
          <div>
            <Section>
              <SectionTitle>Quick Start Tutorial</SectionTitle>
              <VideoContainer>
                <VideoElement controls>
                  <source src="/tutorial-video.mov" type="video/quicktime" />
                  <source src="/tutorial-video.mp4" type="video/mp4" />
                  <source src="/tutorial-video.webm" type="video/webm" />
                  Your browser does not support the video tag.
                </VideoElement>
              </VideoContainer>
            </Section>
            
            <Section>
              <SectionTitle>System Introduction</SectionTitle>
              <Description>
                <p>
                  <strong>InFerActive</strong> is an interactive inference system that visualizes the <strong>sampling space</strong> of local LLM models in tree format. Explore sequential token distributions that appear random but are actually deterministic. Send interaction requests from nodes to trigger real-time backend generation for natural exploration. The system has Llama 3.1 8B FP16 loaded.
                  </p>
              </Description>
            </Section>
            <Section>
              <SectionTitle>Main Goals of the System</SectionTitle>
              <SimpleList>
                <li><strong>Interactive Sampling:</strong> Generate and directly explore LLM sampling space at token level</li>
                <li><strong>Scalable Human Evaluation:</strong> Evaluate LLM model responses per prompt</li>
              </SimpleList>
            </Section>
            <Section>
              <SectionTitle>Usage Scenario</SectionTitle>
              <SimpleList>
                <li><strong>Probabilistic Weighted Evaluation:</strong> Assess the probability of good/bad responses for specific prompts. Identify key branching points and observe response probabilities without generating multiple responses. (e.g., for benchmark prompts with 70% correct answer probability, check distribution in tree instead of generating 10 responses)</li>
                <li><strong>What-if Exploration:</strong> When LLMs generate poor responses with low probability (e.g., jailbreaks), expand the tree from low-probability branches to observe various responses in desired scenarios. (e.g., 95% probability of starting with "No" for unethical requests, but 5% probability of following user requests)</li>
              </SimpleList>
            </Section>
          </div>
          
          <div>
            <Section>
              <SectionTitle>Interface Guide</SectionTitle>
              <ScreenshotContainer>
                <ScreenshotImage 
                  src="/interface-guide.png" 
                  alt="InFerActive Interface Guide"
                />
              </ScreenshotContainer>
              <Description style={{ margin: '16px 0', fontSize: '14px' }}>
                Link thickness in the tree represents token probability. Hover over nodes to display probability tooltips and interaction buttons.
              </Description>
              <AnnotationList>
                <li><strong>Probability Threshold Filter:</strong> Set minimum probability value for displayed nodes. At 20%, only tokens with 20%+ probability are shown.</li>
                <li><strong>Merge/Collapse Button:</strong> Expand or collapse nodes with merged tokens.</li>
                <li><strong>Fold Node:</strong> Collapse this node.</li>
                <li><strong>Fold/Unfold Children:</strong> Expand or collapse all child nodes.</li>
                <li><strong>Token Generation:</strong> Click leaf nodes (unexplored nodes) to start token generation.</li>
              </AnnotationList>
            </Section>
            
            <Section>
              <SectionTitle>⚠️ Error Reporting & Support</SectionTitle>
              <ContactInfo>
                <ContactText>
                  If you encounter any issues using the system, please contact <a >https://github.com/InFerActiveCode/InFerActive</a> and we will respond as quickly as possible after reviewing and addressing the issue.
                </ContactText>
              </ContactInfo>
            </Section>
          </div>
        </ContentGrid>
        
        <ButtonGroup>
          <Button variant="primary" onClick={handleStartDemo}>
            Start Demo
          </Button>
          <CheckboxContainer>
            <input
              type="checkbox"
              checked={dontShowAgain}
              onChange={(e) => setDontShowAgain(e.target.checked)}
            />
            Don't show again
          </CheckboxContainer>
        </ButtonGroup>
      </ModalContent>
    </ModalOverlay>
  );
};