// .st 最小文本模板：esbuild text-loader 把文件内容作为默认导出字符串
declare module "*.st" {
  const content: string;
  export default content;
}