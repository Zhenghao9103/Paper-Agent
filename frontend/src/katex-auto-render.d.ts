declare module "katex/contrib/auto-render" {
  type Delimiter = {
    left: string;
    right: string;
    display: boolean;
  };

  type AutoRenderOptions = {
    delimiters?: Delimiter[];
    throwOnError?: boolean;
    ignoredTags?: string[];
  };

  export default function renderMathInElement(
    element: HTMLElement,
    options?: AutoRenderOptions,
  ): void;
}
