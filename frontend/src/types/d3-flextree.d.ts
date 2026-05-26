declare module 'd3-flextree' {
    export interface FlextreeLayout<Datum> {
      nodeSize(size: (node: any) => [number, number]): this;
      spacing(spacing: (a: any, b: any) => number): this;
      (root: any): any;
    }
  
    export function flextree<Datum>(): FlextreeLayout<Datum>;
  }