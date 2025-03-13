Here's an ASCII art diagram showing the multi-process architecture of the system:
                                                                               
                                   +-------------------+
                                   |                   |
                                   | Wikipedia Dump    |
                                   | (Input File)      |
                                   |                   |
                                   +--------+----------+
                                            |
                                            v 
  +-----------------+           +-----------+-----------+
  |                 |           |                       |
  | Prefetch Process+---------->+ Prefetch Queue (200)  |
  |                 |           |                       |
  +-----------------+           +-----------+-----------+
                                            |     
                                            v
                                +-----------+-----------+
                                |                       |
                                | Reader Process        |                      
                                | (Article Distribution) |                                                                                                                                                                                                                                                                  
                                |                       |
                                +-----+--------+--------+                                                                                                                                                                                                                                                                   
                                      |        |                                                                                                                                                                                                                                                                            
          +---------------------------|--------|---------------------------+                                                                                                                                                                                                                                                
          |                           |        |                           |                                                                                                                                                                                                                                                
          v                           v        v                           v                                                                                                                                                                                                                                                
  +-------+---------+       +---------+--+    ++--------+--+      +--------+------+                                                                                                                                                                                                                                         
  |                 |       |            |    |            |      |               |
  | Input Queue     |       | Worker 1   |    | Worker ... |      | Worker 22     |
  |                 |       |            |    |            |      |               |
  +-+-----+-----+---+       +------------+    +------------+      +---------------+
    |     |     |               |                  |                   |
    |     |     |               |                  |                   |
    v     v     v               v                  v                   v
  +-----+-----+------+       +--+---------------+--+------------------+--------+
  |                  |       |                                                  |
  | Worker Processes |       |    Multiple Output Queues (4 Queues, size 30k)   |
  |                  |       |                                                  |
  +------------------+       +---+------------------------+-------------------+-+
                                 |                        |                   |
                                 v                        v                   v 
                          +------+-------+         +------+------+     +------+------+
                          |              |         |             |     |             |
                          | Main Process |         | Batch Queue |     | Dead Letter |
                          | (Collector)  +-------->+ (Optional)  |     | Queue       |
                          |              |         |             |     |             |
                          +------+-------+         +------+------+     +-------------+
                                 |                        |
                                 v                        v
                          +------+-------+         +------+------+
                          |              |         |             |
                          | Output Files |         | Batch Writer|
                          | (.part files)|         | Process     |
                          |              |         |             |
                          +--------------+         +-------------+

  Key Components

  1. Prefetch Process
    - Reads ahead from Wikipedia dump file
    - Pre-calculates article complexity
    - Buffers articles for smooth processing
  2. Reader Process
    - Takes articles from prefetch queue
    - Sorts by complexity and distributes evenly
    - Sends to workers through input queue
  3. Worker Processes (22)
    - Each assigned to a specific output queue
    - Process articles to extract blocks
    - Output to one of multiple output queues
  4. Output Queues (4)
    - Distribute load across multiple queues
    - Reduce contention for better parallelism
    - Each sized to 30,000 entries
  5. Main Process
    - Monitors all queues in a non-blocking way
    - Collects blocks from all output queues
    - Batches blocks for output
  6. Batch Writer (Optional)
    - Handles disk I/O separately
    - Writes batches without blocking main process
  7. Dead Letter Queue
    - Captures failed articles
    - Allows later analysis of failures

  This architecture maximizes the use of your 24 cores by separating reading, processing, and writing, while using multiple queues to reduce contention points.
