import { NextResponse } from 'next/server';
import { promises as fs } from 'fs';
import path from 'path';

interface FileItem {
  id: string;
  name: string;
  type: 'file' | 'folder' | 'image' | 'video' | 'audio';
  children?: FileItem[];
  isExpanded?: boolean;
  parentId?: string;
  level?: number;
  path: string;
}

function getFileType(fileName: string): FileItem['type'] {
  const ext = path.extname(fileName).toLowerCase();
  
  if (['.jpg', '.jpeg', '.png', '.gif', '.svg', '.webp'].includes(ext)) {
    return 'image';
  }
  if (['.mp4', '.avi', '.mov', '.wmv', '.flv'].includes(ext)) {
    return 'video';
  }
  if (['.mp3', '.wav', '.flac', '.aac', '.ogg'].includes(ext)) {
    return 'audio';
  }
  return 'file';
}

async function buildFileTree(dirPath: string, parentId?: string): Promise<FileItem[]> {
  const items: FileItem[] = [];
  
  try {
    const entries = await fs.readdir(dirPath, { withFileTypes: true });
    
    for (const entry of entries) {
      const fullPath = path.join(dirPath, entry.name);
      const relativePath = path.relative(process.cwd(), fullPath);
      
      const item: FileItem = {
        id: Buffer.from(relativePath).toString('base64'),
        name: entry.name,
        type: entry.isDirectory() ? 'folder' : getFileType(entry.name),
        parentId,
        path: relativePath,
        isExpanded: false,
      };
      
      if (entry.isDirectory()) {
        // Recursively build children for directories
        item.children = await buildFileTree(fullPath, item.id);
      }
      
      items.push(item);
    }
  } catch (error) {
    console.error('Error reading directory:', dirPath, error);
  }
  
  return items.sort((a, b) => {
    // Sort folders first, then files
    if (a.type === 'folder' && b.type !== 'folder') return -1;
    if (a.type !== 'folder' && b.type === 'folder') return 1;
    return a.name.localeCompare(b.name);
  });
}

export async function GET() {
  try {
    const vaultPath = path.join(process.cwd(), 'vault');
    
    // Check if vault directory exists
    try {
      await fs.access(vaultPath);
    } catch {
      // Create vault directory if it doesn't exist
      await fs.mkdir(vaultPath, { recursive: true });
    }
    
    const fileTree = await buildFileTree(vaultPath);
    
    return NextResponse.json({ 
      success: true, 
      data: fileTree 
    });
  } catch (error) {
    console.error('Error building file tree:', error);
    return NextResponse.json({ 
      success: false, 
      error: 'Failed to read file system' 
    }, { status: 500 });
  }
} 