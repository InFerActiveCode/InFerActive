// utils/leftAlignedFlextree.ts
import {hierarchy} from 'd3-hierarchy';

const defaults = Object.freeze({
  children: (data: any) => data.children,
  nodeSize: (node: any) => node.data.size,
  spacing: 0,
});

interface LayoutFunction {
  (tree: any): any;
  nodeSize: (arg?: any) => any;
  spacing: (arg?: any) => any;
  children: (arg?: any) => any;
  hierarchy: (treeData: any, children?: any) => any;
}


export default function leftAlignedFlextree(options: any = {}): LayoutFunction {
  const opts = Object.assign({}, defaults, options);
  function accessor(name: string) {
    const opt = (opts as any)[name];
    return typeof opt === 'function' ? opt : () => opt;
  }

  function layout(tree: any) {
    const wtree = wrap(getWrapper(), tree, (node: any) => node.children);
    wtree.update();
    return wtree.data;
  }

  function getFlexNode() {
    const nodeSize = accessor('nodeSize');
    const spacing = accessor('spacing');
    return class FlexNode extends hierarchy.prototype.constructor {
      constructor(data: any) {
        super(data);
      }
      copy() {
        const c = wrap(this.constructor, this, (node: any) => node.children);
        c.each((node: any) => node.data = node.data.data);
        return c;
      }
      get size() { return nodeSize(this); }
      spacing(oNode: any) { return spacing(this, oNode); }
      get nodes() { return this.descendants(); }
      get xSize() { return this.size[0]; }
      get ySize() { return this.size[1]; }
      get top() { return (this as any).y; }
      get bottom() { return (this as any).y + this.ySize; }
      get left() { return (this as any).x - this.xSize / 2; }
      get right() { return (this as any).x + this.xSize / 2; }
      get root() {
        const ancs = this.ancestors();
        return ancs[ancs.length - 1];
      }
      get numChildren() {
        return this.hasChildren ? (this as any).children.length : 0;
      }
      get hasChildren() { return !this.noChildren; }
      get noChildren() { return (this as any).children === null; }
      get firstChild() {
        return this.hasChildren ? (this as any).children[0] : null;
      }
      get lastChild() {
        return this.hasChildren ? (this as any).children[this.numChildren - 1] : null;
      }
      get extents() {
        return ((this as any).children || []).reduce(
          (acc: any, kid: any) => FlexNode.maxExtents(acc, kid.extents),
          this.nodeExtents);
      }
      get nodeExtents() {
        return {
          top: this.top,
          bottom: this.bottom,
          left: this.left,
          right: this.right,
        };
      }
      static maxExtents(e0: any, e1: any) {
        return {
          top: Math.min(e0.top, e1.top),
          bottom: Math.max(e0.bottom, e1.bottom),
          left: Math.min(e0.left, e1.left),
          right: Math.max(e0.right, e1.right),
        };
      }
    };
  }

  function getWrapper() {
    const FlexNode = getFlexNode();
    const nodeSize = accessor('nodeSize');
    const spacing = accessor('spacing');
    return class extends FlexNode {
      public relX: number = 0;
      public prelim: number = 0;
      public shift: number = 0;
      public change: number = 0;
      public lExt: any = this;
      public lExtRelX: number = 0;
      public lThr: any = null;
      public rExt: any = this;
      public rExtRelX: number = 0;
      public rThr: any = null;

      constructor(data: any) {
        super(data);
        Object.assign(this, {
          x: 0, y: 0,
          relX: 0, prelim: 0, shift: 0, change: 0,
          lExt: this, lExtRelX: 0, lThr: null,
          rExt: this, rExtRelX: 0, rThr: null,
        });
      }
      get size() { return nodeSize(this.data); }
      spacing(oNode: any) { return spacing(this.data, oNode.data); }
      get x() { return this.data.x; }
      set x(v) { this.data.x = v; }
      get y() { return this.data.y; }
      set y(v) { this.data.y = v; }
      update() {
        layoutChildren(this);
        resolveX(this);
        return this;
      }
    };
  }

  function wrap(FlexClass: any, treeData: any, children: any) {
    const _wrap = (data: any, parent: any) => {
      const node = new FlexClass(data);
      Object.assign(node, {
        parent,
        depth: parent === null ? 0 : parent.depth + 1,
        height: 0,
        length: 1,
      });
      const kidsData = children(data) || [];
      node.children = kidsData.length === 0 ? null
        : kidsData.map((kd: any) => _wrap(kd, node));
      if (node.children) {
        Object.assign(node, node.children.reduce(
          (hl: any, kid: any) => ({
            height: Math.max(hl.height, kid.height + 1),
            length: hl.length + kid.length,
          }), node
        ));
      }
      return node;
    };
    return _wrap(treeData, null);
  }

  // *** core flextree algorithm with slight modification to positionRoot ***
  const positionRoot = (w: any) => {
    if (w.hasChildren) {
      const k0 = w.firstChild;
      const kf = w.lastChild;
      
      const prelim = k0.prelim + k0.relX;
      
      Object.assign(w, {
        prelim,
        lExt: k0.lExt, lExtRelX: k0.lExtRelX,
        rExt: kf.rExt, rExtRelX: kf.rExtRelX,
      });
    }
  };

  // not changed
  const layoutChildren = (w: any, y = 0) => {
    w.y = y;
    (w.children || []).reduce((acc: any, kid: any) => {
      const [i, lastLows] = acc;
      layoutChildren(kid, w.y + w.ySize);
      const lowY = (i === 0 ? kid.lExt : kid.rExt).bottom;
      if (i !== 0) separate(w, i, lastLows);
      const lows = updateLows(lowY, i, lastLows);
      return [i + 1, lows];
    }, [0, null]);
    shiftChange(w);
    positionRoot(w);
    return w;
  };

  const resolveX = (w: any, prevSum?: any, parentX?: any) => {
    if (typeof prevSum === 'undefined') {
      prevSum = -w.relX - w.prelim;
      parentX = 0;
    }
    const sum = prevSum + w.relX;
    w.relX = sum + w.prelim - parentX;
    w.prelim = 0;
    w.x = parentX + w.relX;
    (w.children || []).forEach((k: any) => resolveX(k, sum, w.x));
    return w;
  };

  const shiftChange = (w: any) => {
    (w.children || []).reduce((acc: any, child: any) => {
      const [lastShiftSum, lastChangeSum] = acc;
      const shiftSum = lastShiftSum + child.shift;
      const changeSum = lastChangeSum + shiftSum + child.change;
      child.relX += changeSum;
      return [shiftSum, changeSum];
    }, [0, 0]);
  };

  const separate = (w: any, i: any, lows: any) => {
    const lSib = w.children[i - 1];
    const curSubtree = w.children[i];
    let rContour = lSib;
    let rSumMods = lSib.relX;
    let lContour = curSubtree;
    let lSumMods = curSubtree.relX;
    let isFirst = true;
    
    while (rContour && lContour) {
      if (rContour.bottom > lows.lowY) lows = lows.next;
      const dist =
        (rSumMods + rContour.prelim) - (lSumMods + lContour.prelim) +
        rContour.xSize / 2 + lContour.xSize / 2 +
        rContour.spacing(lContour);
      if (dist > 0 || (dist < 0 && isFirst)) {
        lSumMods += dist;
        moveSubtree(curSubtree, dist);
        distributeExtra(w, i, lows.index, dist);
      }
      isFirst = false;
      const rightBottom = rContour.bottom;
      const leftBottom = lContour.bottom;
      if (rightBottom <= leftBottom) {
        rContour = nextRContour(rContour);
        if (rContour) rSumMods += rContour.relX;
      }
      if (rightBottom >= leftBottom) {
        lContour = nextLContour(lContour);
        if (lContour) lSumMods += lContour.relX;
      }
    }
    if (!rContour && lContour) setLThr(w, i, lContour, lSumMods);
    else if (rContour && !lContour) setRThr(w, i, rContour, rSumMods);
  };

  // other helpers
  const moveSubtree = (subtree: any, distance: any) => {
    subtree.relX += distance;
    subtree.lExtRelX += distance;
    subtree.rExtRelX += distance;
  };

  const distributeExtra = (w: any, curSubtreeI: any, leftSibI: any, dist: any) => {
    const curSubtree = w.children[curSubtreeI];
    const n = curSubtreeI - leftSibI;
    if (n > 1) {
      const delta = dist / n;
      w.children[leftSibI + 1].shift += delta;
      curSubtree.shift -= delta;
      curSubtree.change -= dist - delta;
    }
  };

  const nextLContour = (w: any) => {
    return w.hasChildren ? w.firstChild : w.lThr;
  };

  const nextRContour = (w: any) => {
    return w.hasChildren ? w.lastChild : w.rThr;
  };

  const setLThr = (w: any, i: any, lContour: any, lSumMods: any) => {
    const firstChild = w.firstChild;
    const lExt = firstChild.lExt;
    const curSubtree = w.children[i];
    lExt.lThr = lContour;
    const diff = lSumMods - lContour.relX - firstChild.lExtRelX;
    lExt.relX += diff;
    lExt.prelim -= diff;
    firstChild.lExt = curSubtree.lExt;
    firstChild.lExtRelX = curSubtree.lExtRelX;
  };

  const setRThr = (w: any, i: any, rContour: any, rSumMods: any) => {
    const curSubtree = w.children[i];
    const rExt = curSubtree.rExt;
    const lSib = w.children[i - 1];
    rExt.rThr = rContour;
    const diff = rSumMods - rContour.relX - curSubtree.rExtRelX;
    rExt.relX += diff;
    rExt.prelim -= diff;
    curSubtree.rExt = lSib.rExt;
    curSubtree.rExtRelX = lSib.rExtRelX;
  };

  const updateLows = (lowY: any, index: any, lastLows: any) => {
    while (lastLows !== null && lowY >= lastLows.lowY)
      lastLows = lastLows.next;
    return {
      lowY,
      index,
      next: lastLows,
    };
  };

  Object.assign(layout, {
    nodeSize(arg?: any) {
      return arguments.length ? ((opts as any).nodeSize = arg, layout) : (opts as any).nodeSize;
    },
    spacing(arg?: any) {
      return arguments.length ? ((opts as any).spacing = arg, layout) : (opts as any).spacing;
    },
    children(arg?: any) {
      return arguments.length ? ((opts as any).children = arg, layout) : (opts as any).children;
    },
    hierarchy(treeData: any, children?: any) {
      const kids = typeof children === 'undefined' ? (opts as any).children : children;
      return wrap(getFlexNode(), treeData, kids);
    },
  });
  
  return layout as LayoutFunction;
}