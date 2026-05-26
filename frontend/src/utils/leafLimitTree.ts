import { compareNodesByProbability, selectTopNLikeLeafNodes } from '../components/tokenTreeUtils';
import { VisualNode } from '../types/types';

export const LEAF_LIMIT_OPTIONS = [10, 20, 30, 50] as const;

export function createLeafLimitedTree(root: VisualNode, targetLeafCount: number): VisualNode {
  const getValidChildren = (node: VisualNode): VisualNode[] => (
    [...node.children].sort(compareNodesByProbability)
  );

  const selectedLeafNodes = selectTopNLikeLeafNodes(root, targetLeafCount, getValidChildren);
  const selectedLeafIds = new Set(selectedLeafNodes.map(node => node.id));

  const pruneNode = (node: VisualNode): VisualNode | null => {
    const keptChildren = node.children
      .map(pruneNode)
      .filter((child): child is VisualNode => child !== null);

    if (node.id !== root.id && !selectedLeafIds.has(node.id) && keptChildren.length === 0) {
      return null;
    }

    return {
      ...node,
      children: keptChildren,
    };
  };

  return pruneNode(root) ?? { ...root, children: [] };
}
