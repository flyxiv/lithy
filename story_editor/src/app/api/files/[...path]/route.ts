import { NextRequest, NextResponse } from 'next/server';
import { promises as fs } from 'fs';
import path from 'path';

export async function GET(
  request: NextRequest,
  { params }: { params: { path: string[] } }
) {
  try {
    const filePath = params.path.join('/');
    const fullPath = path.join(process.cwd(), 'vault', filePath);
    
    // Security check: ensure the path is within the vault directory
    const vaultPath = path.join(process.cwd(), 'vault');
    if (!fullPath.startsWith(vaultPath)) {
      return NextResponse.json({ 
        success: false, 
        error: 'Access denied' 
      }, { status: 403 });
    }
    
    // Check if file exists and is a file (not directory)
    const stats = await fs.stat(fullPath);
    if (!stats.isFile()) {
      return NextResponse.json({ 
        success: false, 
        error: 'Not a file' 
      }, { status: 400 });
    }
    
    // Read file content
    const content = await fs.readFile(fullPath, 'utf-8');
    
    return NextResponse.json({ 
      success: true, 
      data: {
        content,
        path: filePath,
        size: stats.size,
        modified: stats.mtime.toISOString()
      }
    });
  } catch (error) {
    console.error('Error reading file:', error);
    return NextResponse.json({ 
      success: false, 
      error: 'Failed to read file' 
    }, { status: 500 });
  }
}

export async function PUT(
  request: NextRequest,
  { params }: { params: { path: string[] } }
) {
  try {
    const filePath = params.path.join('/');
    const fullPath = path.join(process.cwd(), 'vault', filePath);
    
    // Security check: ensure the path is within the vault directory
    const vaultPath = path.join(process.cwd(), 'vault');
    if (!fullPath.startsWith(vaultPath)) {
      return NextResponse.json({ 
        success: false, 
        error: 'Access denied' 
      }, { status: 403 });
    }
    
    const { content } = await request.json();
    
    // Ensure directory exists
    const dir = path.dirname(fullPath);
    await fs.mkdir(dir, { recursive: true });
    
    // Write file content
    await fs.writeFile(fullPath, content, 'utf-8');
    
    return NextResponse.json({ 
      success: true, 
      message: 'File saved successfully' 
    });
  } catch (error) {
    console.error('Error saving file:', error);
    return NextResponse.json({ 
      success: false, 
      error: 'Failed to save file' 
    }, { status: 500 });
  }
} 