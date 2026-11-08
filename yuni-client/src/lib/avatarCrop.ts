import type { Area } from 'react-easy-crop'

export function validateAvatarFile(file: File) {
  if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) throw new Error('请选择 JPEG、PNG 或 WebP 图片')
  if (!file.size || file.size > 5 * 1024 * 1024) throw new Error('请选择不超过 5 MB 的有效图片')
}

export function loadAvatarImage(source: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image()
    image.onload = () => {
      if (!image.naturalWidth || !image.naturalHeight || image.naturalWidth * image.naturalHeight > 100_000_000) reject(new Error('图片尺寸无效或分辨率过大'))
      else resolve(image)
    }
    image.onerror = () => reject(new Error('图片无法读取，请换一张图片'))
    image.src = source
  })
}

export async function cropAvatar(image: HTMLImageElement, area: Area, name: string): Promise<File> {
  const x = Math.max(0, area.x), y = Math.max(0, area.y)
  const width = Math.min(area.width, image.naturalWidth - x)
  const height = Math.min(area.height, image.naturalHeight - y)
  if (![x, y, width, height].every(Number.isFinite) || width <= 0 || height <= 0) throw new Error('请重新选择裁剪区域')
  const canvas = document.createElement('canvas')
  canvas.width = 512
  canvas.height = 512
  const context = canvas.getContext('2d')
  if (!context) throw new Error('当前环境无法裁剪图片')
  context.imageSmoothingQuality = 'high'
  context.drawImage(image, x, y, width, height, 0, 0, 512, 512)
  const blob = await new Promise<Blob>((resolve, reject) => canvas.toBlob(
    (result) => result ? resolve(result) : reject(new Error('裁剪图片生成失败')), 'image/png',
  ))
  return new File([blob], `${name.replace(/\.[^.]+$/, '') || 'avatar'}-cropped.png`, { type: 'image/png' })
}
