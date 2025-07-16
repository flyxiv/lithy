'use client';

import { useState, useEffect } from 'react';
import { useEditor, EditorContent } from '@tiptap/react';
import StarterKit from '@tiptap/starter-kit';
import { 
  FaFileAlt, 
  FaImage, 
  FaFolder, 
  FaPlus, 
  FaChevronRight, 
  FaChevronDown,
  FaFile,
  FaVideo,
  FaMusic,
  FaGripVertical
} from 'react-icons/fa';
import {
  DndContext,
  closestCenter,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
  DragEndEvent,
  DragOverlay,
  DragStartEvent,
} from '@dnd-kit/core';
import {
  SortableContext,
  sortableKeyboardCoordinates,
  verticalListSortingStrategy,
  useSortable,
} from '@dnd-kit/sortable';
import {
  CSS,
} from '@dnd-kit/utilities';

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

// Sortable File Item Component
function SortableFileItem({ 
  item, 
  level, 
  onToggle, 
  onSelect, 
  selectedFile 
}: {
  item: FileItem;
  level: number;
  onToggle: (id: string) => void;
  onSelect: (file: FileItem) => void;
  selectedFile: FileItem | null;
}) {
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging,
  } = useSortable({ id: item.id });

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.5 : 1,
  };

  const getFileIcon = (type: string) => {
    switch (type) {
      case 'file':
        return <FaFileAlt className="text-blue-600" />;
      case 'folder':
        return <FaFolder className="text-yellow-600" />;
      case 'image':
        return <FaImage className="text-green-600" />;
      case 'video':
        return <FaVideo className="text-purple-600" />;
      case 'audio':
        return <FaMusic className="text-red-600" />;
      default:
        return <FaFile className="text-gray-600" />;
    }
  };

  return (
    <div 
      ref={setNodeRef}
      style={style}
      className={`ml-${level * 4}`}
    >
      <div 
        className={`flex items-center space-x-2 py-1 px-2 rounded cursor-pointer hover:bg-gray-100 group ${
          selectedFile?.id === item.id ? 'bg-blue-50 text-blue-700' : ''
        }`}
        onClick={() => {
          if (item.type === 'folder') {
            onToggle(item.id);
          } else {
            onSelect(item);
          }
        }}
      >
        {/* Drag Handle */}
        <div
          {...attributes}
          {...listeners}
          className="opacity-0 group-hover:opacity-100 transition-opacity cursor-grab active:cursor-grabbing"
        >
          <FaGripVertical className="text-gray-400" size={12} />
        </div>
        
        {/* Folder Chevron */}
        {item.type === 'folder' && (
          <span className="text-gray-500">
            {item.isExpanded ? <FaChevronDown size={12} /> : <FaChevronRight size={12} />}
          </span>
        )}
        
        {/* File Icon */}
        <span className="text-sm">
          {getFileIcon(item.type)}
        </span>
        
        {/* File Name */}
        <span className="text-sm text-gray-700">{item.name}</span>
      </div>
    </div>
  );
}

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<FileItem | null>(null);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [editorState, setEditorState] = useState<any>(null);
  const [isClient, setIsClient] = useState(false);
  const [fileTree, setFileTree] = useState<FileItem[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isLoadingFile, setIsLoadingFile] = useState(false);

  const sensors = useSensors(
    useSensor(PointerSensor),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    })
  );

  const editor = useEditor({
    extensions: [StarterKit],
    content: `
      <h1>Chapter 1: The Beginning</h1>
      <p>Emily stood at the crossroads, her heart heavy with the weight of her past decisions. The war had changed everything, and she couldn't shake the feeling that she had abandoned those who needed her most.</p>
      
      <h2>Character Background</h2>
      <p>As a Private World Investigator, Emily had seen the worst of humanity. Her childhood experiences during the war had shaped her into someone who valued independence above all else, yet deep down, she yearned for the connections she had severed.</p>
      
      <blockquote>
        <p>"Sometimes the hardest person to forgive is yourself."</p>
      </blockquote>
      
      <p>The investigation she was about to undertake would force her to confront not just external mysteries, but the ghosts of her own past...</p>
    `,
    editorProps: {
      attributes: {
        class: 'focus:outline-none min-h-[500px] p-4 text-gray-900 leading-relaxed',
      },
    },
    immediatelyRender: false,
    onUpdate: ({ editor }) => {
      // Update editor state to trigger toolbar re-render
      setEditorState(editor.state);
    },
    onFocus: ({ editor }) => {
      // Update state when editor gets focus
      setEditorState(editor.state);
    },
    onSelectionUpdate: ({ editor }) => {
      // Update state when selection changes
      setEditorState(editor.state);
    },
  });

  // Initialize client-side rendering
  useEffect(() => {
    setIsClient(true);
  }, []);

  // Load file tree from file system
  useEffect(() => {
    const loadFileTree = async () => {
      try {
        setIsLoading(true);
        const response = await fetch('/api/files');
        const result = await response.json();
        
        if (result.success) {
          setFileTree(result.data);
        } else {
          console.error('Failed to load file tree:', result.error);
        }
      } catch (error) {
        console.error('Error loading file tree:', error);
      } finally {
        setIsLoading(false);
      }
    };
    
    loadFileTree();
  }, []);

  // Initialize editor state after editor is ready
  useEffect(() => {
    if (editor && !editorState) {
      setEditorState(editor.state);
      console.log('Editor initialized:', editor);
      console.log('Available commands:', editor.commands);
    }
  }, [editor, editorState]);

  // Load file content
  const loadFileContent = async (file: FileItem) => {
    if (file.type === 'folder') return;
    
    try {
      setIsLoadingFile(true);
      
      // Extract path relative to vault
      const relativePath = file.path.replace('vault/', '');
      const response = await fetch(`/api/files/${relativePath}`);
      const result = await response.json();
      
      if (result.success && editor) {
        // Set content in editor
        editor.commands.setContent(result.data.content);
        setSelectedFile(file);
      } else {
        console.error('Failed to load file:', result.error);
      }
    } catch (error) {
      console.error('Error loading file:', error);
    } finally {
      setIsLoadingFile(false);
    }
  };

  // Save file content
  const saveFileContent = async () => {
    if (!selectedFile || !editor) return;
    
    try {
      const content = editor.getHTML();
      const relativePath = selectedFile.path.replace('vault/', '');
      
      const response = await fetch(`/api/files/${relativePath}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content })
      });
      
      const result = await response.json();
      
      if (result.success) {
        console.log('File saved successfully');
        // TODO: Show success message
      } else {
        console.error('Failed to save file:', result.error);
      }
    } catch (error) {
      console.error('Error saving file:', error);
    }
  };

  const flattenFileTree = (items: FileItem[]): FileItem[] => {
    const result: FileItem[] = [];
    
    function traverse(nodes: FileItem[], level: number = 0) {
      for (const node of nodes) {
        result.push({ ...node, level });
        if (node.children && node.isExpanded) {
          traverse(node.children, level + 1);
        }
      }
    }
    
    traverse(items);
    return result;
  };

  const handleDragStart = (event: DragStartEvent) => {
    const { active } = event;
    setActiveId(active.id as string);
  };

  const handleDragEnd = (event: DragEndEvent) => {
    const { active, over } = event;
    
    if (!over || active.id === over.id) {
      setActiveId(null);
      return;
    }

    setFileTree((items) => {
      const flatItems = flattenFileTree(items);
      const oldIndex = flatItems.findIndex((item) => item.id === active.id);
      const newIndex = flatItems.findIndex((item) => item.id === over.id);
      
      // Simple reordering for demonstration
      // In a real app, you'd need more complex logic to handle parent-child relationships
      const newItems = [...items];
      // This is a simplified implementation - you would need more complex logic
      // to handle proper tree restructuring
      
      return newItems;
    });
    
    setActiveId(null);
  };

  const toggleFolder = (id: string) => {
    setFileTree(prev => {
      const updateItem = (items: FileItem[]): FileItem[] => {
        return items.map(item => {
          if (item.id === id) {
            return { ...item, isExpanded: !item.isExpanded };
          }
          if (item.children) {
            return { ...item, children: updateItem(item.children) };
          }
          return item;
        });
      };
      return updateItem(prev);
    });
  };

  const addNewFile = () => {
    const fileName = prompt('Enter file name:');
    if (fileName) {
      const newFile: FileItem = {
        id: Date.now().toString(),
        name: fileName,
        type: fileName.includes('.') ? 'file' : 'folder',
        path: `vault/${fileName}`
      };
      
      setFileTree(prev => [
        ...prev,
        newFile
      ]);
      
      // TODO: Actually create the file on the server
      // This would need another API endpoint
    }
  };

  const renderFileTree = (items: FileItem[], level: number = 0) => {
    const flatItems = flattenFileTree(items);
    
    if (!isClient) {
      // Server-side rendering: render simple file tree without drag & drop
      return (
        <div className="space-y-1">
          {flatItems.map((item) => (
            <div key={item.id} className={`ml-${level * 4}`}>
              <div 
                                 className={`flex items-center space-x-2 py-1 px-2 rounded cursor-pointer hover:bg-gray-100 ${
                   selectedFile?.id === item.id ? 'bg-blue-50 text-blue-700' : ''
                 }`}
                 onClick={() => {
                   if (item.type === 'folder') {
                     toggleFolder(item.id);
                   } else {
                     loadFileContent(item);
                   }
                 }}
              >
                {item.type === 'folder' && (
                  <span className="text-gray-500">
                    {item.isExpanded ? <FaChevronDown size={12} /> : <FaChevronRight size={12} />}
                  </span>
                )}
                <span className="text-sm">
                  {item.type === 'file' ? <FaFileAlt className="text-blue-600" /> :
                   item.type === 'folder' ? <FaFolder className="text-yellow-600" /> :
                   item.type === 'image' ? <FaImage className="text-green-600" /> :
                   item.type === 'video' ? <FaVideo className="text-purple-600" /> :
                   item.type === 'audio' ? <FaMusic className="text-red-600" /> :
                   <FaFile className="text-gray-600" />}
                </span>
                <span className="text-sm text-gray-700">{item.name}</span>
              </div>
            </div>
          ))}
        </div>
      );
    }
    
    return (
      <SortableContext items={flatItems.map(item => item.id)} strategy={verticalListSortingStrategy}>
        {flatItems.map((item) => (
          <SortableFileItem
            key={item.id}
            item={item}
            level={level}
            onToggle={toggleFolder}
            onSelect={loadFileContent}
            selectedFile={selectedFile}
          />
        ))}
      </SortableContext>
    );
  };

  const FileExplorer = () => (
    <div className="w-80 bg-white shadow-md">
      {/* Header with Add Button */}
      <div className="flex items-center justify-between p-4 border-b border-gray-200">
        <h1 className="text-lg font-semibold text-gray-800">Story Editor</h1>
        <button
          onClick={addNewFile}
          className="p-2 text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded-lg transition-colors"
          title="Add new file"
        >
          <FaPlus size={16} />
        </button>
      </div>

      {/* File Tree */}
      <div className="p-4 overflow-y-auto">
        <div className="space-y-1">
          {isLoading ? (
            <div className="text-gray-500 text-sm">Loading files...</div>
          ) : fileTree.length === 0 ? (
            <div className="text-gray-500 text-sm">No files found</div>
          ) : (
            renderFileTree(fileTree)
          )}
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-50 flex">
      {isClient ? (
        <DndContext
          sensors={sensors}
          collisionDetection={closestCenter}
          onDragStart={handleDragStart}
          onDragEnd={handleDragEnd}
        >
          <FileExplorer />
          {/* Drag Overlay */}
          <DragOverlay>
            {activeId ? (
              <div className="bg-white shadow-lg rounded px-2 py-1 border">
                <span className="text-sm text-gray-700">
                  {fileTree.find(item => item.id === activeId)?.name}
                </span>
              </div>
            ) : null}
          </DragOverlay>
        </DndContext>
      ) : (
        <FileExplorer />
      )}

      {/* Main Content - Text Editor */}
      <div className="flex-1 flex flex-col">
        {/* Editor Header */}
        <div className="bg-white border-b border-gray-200 p-4">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-medium text-gray-800">
              {selectedFile?.name || 'Select a file to edit'}
            </h2>
            <div className="flex items-center space-x-2">
              <button 
                onClick={saveFileContent}
                disabled={!selectedFile || isLoadingFile}
                className="px-3 py-1 text-sm text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded disabled:opacity-50"
              >
                Save
              </button>
              <button className="px-3 py-1 text-sm text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded">
                Export
              </button>
            </div>
          </div>
        </div>

        {/* Editor Toolbar */}
        <div className="bg-white border-b border-gray-200 p-3">
          {!editor ? (
            <div className="text-gray-500">Loading editor...</div>
          ) : (
            <div className="flex items-center space-x-1" key={editorState?.doc.content.size}>
            <button
              onClick={() => editor?.chain().focus().toggleBold().run()}
              className={`px-3 py-2 rounded-md text-sm font-medium border transition-colors ${
                editor?.isActive('bold') 
                  ? 'bg-blue-500 text-white border-blue-500 shadow-sm' 
                  : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-50 hover:border-gray-400'
              }`}
            >
              <strong>B</strong>
            </button>
            <button
              onClick={() => editor?.chain().focus().toggleItalic().run()}
              className={`px-3 py-2 rounded-md text-sm font-medium border transition-colors ${
                editor?.isActive('italic') 
                  ? 'bg-blue-500 text-white border-blue-500 shadow-sm' 
                  : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-50 hover:border-gray-400'
              }`}
            >
              <em>I</em>
            </button>
            <button
              onClick={() => editor?.chain().focus().toggleStrike().run()}
              className={`px-3 py-2 rounded-md text-sm font-medium border transition-colors ${
                editor?.isActive('strike') 
                  ? 'bg-blue-500 text-white border-blue-500 shadow-sm' 
                  : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-50 hover:border-gray-400'
              }`}
            >
              <s>S</s>
            </button>
            <div className="w-px h-8 bg-gray-300 mx-2"></div>
            <button
              onClick={() => {
                console.log('H1 clicked, editor:', editor);
                if (editor) {
                  editor.chain().focus().toggleHeading({ level: 1 }).run();
                  console.log('H1 is active:', editor.isActive('heading', { level: 1 }));
                }
              }}
              className={`px-3 py-2 rounded-md text-sm font-medium border transition-colors ${
                editor?.isActive('heading', { level: 1 }) 
                  ? 'bg-blue-500 text-white border-blue-500 shadow-sm' 
                  : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-50 hover:border-gray-400'
              }`}
            >
              H1
            </button>
            <button
              onClick={() => {
                console.log('H2 clicked, editor:', editor);
                if (editor) {
                  editor.chain().focus().toggleHeading({ level: 2 }).run();
                  console.log('H2 is active:', editor.isActive('heading', { level: 2 }));
                }
              }}
              className={`px-3 py-2 rounded-md text-sm font-medium border transition-colors ${
                editor?.isActive('heading', { level: 2 }) 
                  ? 'bg-blue-500 text-white border-blue-500 shadow-sm' 
                  : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-50 hover:border-gray-400'
              }`}
            >
              H2
            </button>
            <button
              onClick={() => {
                console.log('Quote clicked, editor:', editor);
                if (editor) {
                  editor.chain().focus().toggleBlockquote().run();
                  console.log('Quote is active:', editor.isActive('blockquote'));
                }
              }}
              className={`px-3 py-2 rounded-md text-sm font-medium border transition-colors ${
                editor?.isActive('blockquote') 
                  ? 'bg-blue-500 text-white border-blue-500 shadow-sm' 
                  : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-50 hover:border-gray-400'
              }`}
            >
              Quote
            </button>
          </div>
          )}
        </div>

        {/* Editor Content */}
        <div className="flex-1 bg-white">
          {isLoadingFile ? (
            <div className="p-4 text-gray-500">Loading file...</div>
          ) : editor ? (
            <EditorContent 
              editor={editor} 
              className="h-full prose-editor"
            />
          ) : (
            <div className="p-4 text-gray-500">Loading editor...</div>
          )}
        </div>
      </div>

      {/* Right Sidebar - AI Feedback */}
      <div className="w-80 bg-white shadow-md p-6">
        <h2 className="text-xl font-light text-gray-800 mb-6">AI Feedback</h2>
        
        <div className="space-y-4">
          <div className="bg-gray-50 p-4 rounded-lg">
            <p className="text-sm text-gray-600 mb-2">Writing Style</p>
            <p className="text-xs text-gray-500 leading-relaxed">
              Your narrative voice is compelling. The emotional depth in Emily's character development shows strong storytelling potential.
            </p>
          </div>
          
          <div className="bg-gray-50 p-4 rounded-lg">
            <p className="text-sm text-gray-600 mb-2">Pacing Analysis</p>
            <p className="text-xs text-gray-500 leading-relaxed">
              Consider varying sentence length more to create rhythm. The current pacing works well for introspective scenes.
            </p>
          </div>
          
          <div className="bg-gray-50 p-4 rounded-lg">
            <p className="text-sm text-gray-600 mb-2">Character Development</p>
            <p className="text-xs text-gray-500 leading-relaxed">
              Emily's internal conflict is well-established. Adding more specific memories could deepen reader connection.
            </p>
          </div>

          <div className="bg-gray-50 p-4 rounded-lg">
            <p className="text-sm text-gray-600 mb-2">Dialogue Suggestions</p>
            <p className="text-xs text-gray-500 leading-relaxed">
              The quote you've included is impactful. Consider adding more dialogue to break up exposition.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
