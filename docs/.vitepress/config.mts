import { defineConfig } from 'vitepress'

export default defineConfig({
  title: '一份笔记',
  description: '尽量记录学过的东西',
  cleanUrls: true,
  ignoreDeadLinks: true,
  srcDir: '..',
  srcExclude: [
    'docs/**',
    'assets/**',
    'scripts/**',
    '_templates/**',
    '_不上传的暂存/**',
    'node_modules/**',
    '**/activate_functions.png',
  ],
  themeConfig: {
    nav: [
      { text: 'AI', link: '/ai/' },
      { text: '嵌入式', link: '/嵌入式/' },
      { text: '其他', link: '/其他/' },
      { text: 'Linux', link: '/linux/' },
    ],
    sidebar: {
      '/ai/': [
        {
          text: 'AI',
          items: [
            { text: '总览', link: '/ai/' },
            { text: 'MiniMind 大模型入门笔记', link: '/ai/MiniMind——大模型入门笔记' },
            {
              text: 'MiniMind 手撕',
              items: [
                { text: 'Transformer', link: '/ai/minimind/transformer' },
                { text: 'RMSNorm', link: '/ai/minimind/RMSNorm' },
                { text: 'Attention（GQA）', link: '/ai/minimind/attention' },
                { text: 'FFN', link: '/ai/minimind/ffn' },
                { text: 'RoPE + YaRN', link: '/ai/minimind/RoPE+YaRN' },
                { text: '激活函数', link: '/ai/minimind/激活函数' },
                { text: '预训练 + 微调', link: '/ai/minimind/预训练+微调' },
              ],
            },
          ],
        },
      ],
      '/嵌入式/': [
        {
          text: '嵌入式',
          items: [{ text: '总览', link: '/嵌入式/' }],
        },
      ],
      '/其他/': [
        {
          text: '其他',
          items: [{ text: '总览', link: '/其他/' }],
        },
      ],
      '/linux/': [
        {
          text: 'Linux',
          items: [{ text: '总览', link: '/linux/' }],
        },
      ],
    },
    search: {
      provider: 'local',
    },
  },
})
