"""
Background processing service for heavy operations.
Handles vector operations, document processing, and other CPU-intensive tasks.
"""

import logging
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BackgroundTask:
    task_id: str
    name: str
    function: Callable
    args: tuple
    kwargs: dict
    status: TaskStatus
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Any = None
    error: Optional[str] = None
    progress: int = 0  # 0-100 percentage


class BackgroundProcessor:
    """
    Background processor for heavy operations with progress tracking.
    Uses ThreadPoolExecutor for CPU-intensive tasks.
    """
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="BgProcessor")
        self.tasks: Dict[str, BackgroundTask] = {}
        self.task_queue = queue.Queue()
        self._shutdown = False
        
        # Start background monitor thread
        self.monitor_thread = threading.Thread(target=self._monitor_tasks, daemon=True)
        self.monitor_thread.start()
        
        logger.info(f"Background processor initialized with {max_workers} workers")
    
    def submit_task(self, name: str, function: Callable, *args, **kwargs) -> str:
        """
        Submit a task for background processing.
        
        Args:
            name: Human-readable task name
            function: Function to execute
            *args, **kwargs: Arguments for the function
            
        Returns:
            task_id: Unique identifier for the task
        """
        task_id = str(uuid.uuid4())
        task = BackgroundTask(
            task_id=task_id,
            name=name,
            function=function,
            args=args,
            kwargs=kwargs,
            status=TaskStatus.PENDING,
            created_at=time.time()
        )
        
        self.tasks[task_id] = task
        
        # Submit to executor
        future = self.executor.submit(self._execute_task, task)
        
        logger.info(f"Submitted background task: {name} [{task_id}]")
        return task_id
    
    def _execute_task(self, task: BackgroundTask) -> Any:
        """Execute a background task with error handling and progress tracking."""
        try:
            task.status = TaskStatus.RUNNING
            task.started_at = time.time()
            
            logger.debug(f"Starting task: {task.name} [{task.task_id}]")
            
            # Execute the function
            result = task.function(*task.args, **task.kwargs)
            
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = time.time()
            task.progress = 100
            
            duration = task.completed_at - task.started_at
            logger.info(f"Completed task: {task.name} [{task.task_id}] in {duration:.2f}s")
            
            return result
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = time.time()
            
            logger.error(f"Task failed: {task.name} [{task.task_id}] - {e}")
            raise
    
    def get_task_status(self, task_id: str) -> Optional[BackgroundTask]:
        """Get the status of a specific task."""
        return self.tasks.get(task_id)
    
    def get_all_tasks(self) -> Dict[str, BackgroundTask]:
        """Get all tasks and their statuses."""
        return self.tasks.copy()
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a pending task.
        Note: Running tasks cannot be cancelled safely.
        """
        task = self.tasks.get(task_id)
        if not task:
            return False
        
        if task.status == TaskStatus.PENDING:
            task.status = TaskStatus.CANCELLED
            task.completed_at = time.time()
            logger.info(f"Cancelled task: {task.name} [{task_id}]")
            return True
        
        return False
    
    def _monitor_tasks(self):
        """Monitor tasks and clean up completed ones."""
        while not self._shutdown:
            try:
                current_time = time.time()
                tasks_to_remove = []
                
                # Clean up old completed/failed tasks (older than 1 hour)
                for task_id, task in self.tasks.items():
                    if (task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED] 
                        and task.completed_at 
                        and current_time - task.completed_at > 3600):  # 1 hour
                        tasks_to_remove.append(task_id)
                
                # Remove old tasks
                for task_id in tasks_to_remove:
                    del self.tasks[task_id]
                    logger.debug(f"Cleaned up old task: {task_id}")
                
                # Sleep for 60 seconds before next cleanup
                time.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in task monitor: {e}")
                time.sleep(10)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics."""
        current_time = time.time()
        
        status_counts = {}
        for status in TaskStatus:
            status_counts[status.value] = 0
        
        recent_tasks = 0  # Tasks in last hour
        
        for task in self.tasks.values():
            status_counts[task.status.value] += 1
            if current_time - task.created_at < 3600:  # Last hour
                recent_tasks += 1
        
        return {
            'total_tasks': len(self.tasks),
            'recent_tasks_1h': recent_tasks,
            'max_workers': self.max_workers,
            'active_threads': threading.active_count(),
            'status_counts': status_counts
        }
    
    def shutdown(self):
        """Shutdown the background processor gracefully."""
        logger.info("Shutting down background processor...")
        self._shutdown = True
        
        # Wait for running tasks to complete (with timeout)
        self.executor.shutdown(wait=True, timeout=30)
        
        logger.info("Background processor shutdown complete")


# Specific background task functions for the application
class DocumentProcessingTasks:
    """Collection of document processing tasks optimized for background execution."""
    
    @staticmethod
    def process_document_embeddings(document_chunks, embedding_model):
        """Process document embeddings in background."""
        try:
            embeddings = []
            total_chunks = len(document_chunks)
            
            logger.info(f"Processing {total_chunks} document chunks for embeddings")
            
            # Process in batches for better memory management
            batch_size = 10
            for i in range(0, total_chunks, batch_size):
                batch = document_chunks[i:i + batch_size]
                batch_texts = [chunk.content for chunk in batch]
                
                # Generate embeddings for batch
                if embedding_model:
                    batch_embeddings = embedding_model.embed_documents(batch_texts)
                    embeddings.extend(batch_embeddings)
                
                # Progress update (this could be enhanced with a callback)
                progress = int((i + len(batch)) / total_chunks * 100)
                logger.debug(f"Embedding progress: {progress}%")
            
            logger.info(f"Successfully processed embeddings for {len(embeddings)} chunks")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error processing document embeddings: {e}")
            raise
    
    @staticmethod
    def bulk_vector_search(query_embeddings, vector_ops, top_k=20):
        """Perform bulk vector search operations in background."""
        try:
            results = []
            total_queries = len(query_embeddings)
            
            logger.info(f"Processing {total_queries} vector search queries")
            
            for i, query_embedding in enumerate(query_embeddings):
                query_results = vector_ops.search_similar_chunks(query_embedding, top_k)
                results.append(query_results)
                
                progress = int((i + 1) / total_queries * 100)
                logger.debug(f"Search progress: {progress}%")
            
            logger.info(f"Completed {len(results)} vector searches")
            return results
            
        except Exception as e:
            logger.error(f"Error in bulk vector search: {e}")
            raise


# Global background processor instance
_background_processor = None

def get_background_processor() -> BackgroundProcessor:
    """Get the global background processor instance."""
    global _background_processor
    if _background_processor is None:
        _background_processor = BackgroundProcessor()
    return _background_processor

def shutdown_background_processor():
    """Shutdown the global background processor."""
    global _background_processor
    if _background_processor:
        _background_processor.shutdown()
        _background_processor = None