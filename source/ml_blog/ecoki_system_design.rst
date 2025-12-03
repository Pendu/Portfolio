.. _ecoki_system_design:

================================================================================
EcoKI System Design: Architecting a Low-Code ML Platform for Industrial IoT
================================================================================

.. figure:: ecoki_architecture_files/Gemini_Generated_Ecoki_architecture.png
   :alt: EcoKI System Architecture - A comprehensive view of the layered microservices architecture
   :align: center
   :width: 100%
   :figclass: align-center

   **EcoKI System Architecture**: A layered microservices architecture designed for industrial machine learning applications. The system comprises five distinct layers: Frontend, Backend, Execution Engine, Resource Pool, and Data Acquisition & Storage.

How do you design a machine learning platform that can serve diverse industrial use cases, from energy optimization in factories to predictive maintenance in production lines, while remaining accessible to engineers without deep ML expertise? This was the central design challenge behind **EcoKI**, a low-code ML platform I helped architect and build.

This post offers a deep technical dive into the system design decisions that shaped EcoKI. We'll explore each architectural layer, examine the design patterns that enable modularity and scalability, and discuss the trade-offs inherent in building production-grade ML systems for industrial environments.

.. contents:: Table of Contents
   :local:
   :depth: 2

Architectural Philosophy: Design Principles
===========================================

Before diving into the specifics, it's essential to understand the core design principles that guided our architectural decisions.

**Separation of Concerns**

The architecture strictly separates concerns across five distinct layers:

1. **Presentation Layer** (Frontend) — User interaction and visualization
2. **API Gateway Layer** (Backend) — Request routing and orchestration
3. **Execution Layer** (Execution Engine) — Computation and ML pipeline processing
4. **Configuration Layer** (Resource Pool) — Static assets and metadata management
5. **Data Layer** (Data Acquisition & Storage) — Industrial data ingestion and persistence

This separation allows independent scaling, deployment, and maintenance of each component.

**Stateless Building Blocks**

Perhaps the most critical design decision was making all **Building Blocks stateless**. A Building Block is a self-contained unit of computation (e.g., data imputation, model training, feature selection) that:

- Takes inputs through well-defined ports
- Performs a single, atomic operation
- Produces outputs without retaining internal state between executions

.. note::

   **Why Statelessness?** Stateless components are inherently easier to test, parallelize, and scale horizontally. They eliminate race conditions and make debugging deterministic.

**Executor Pattern for Lifecycle Management**

Since Building Blocks are stateless, we introduced the **Executor Pattern** to manage their lifecycle. Executors handle:

- Resource allocation and cleanup
- Error handling and retry logic
- Logging and telemetry
- State persistence (externalized from the blocks themselves)

This separation of *what* to compute (Building Blocks) from *how* to execute (Executors) is a key enabler of the platform's flexibility.

Layer 1: Frontend — The Client Interface
=========================================

The frontend layer provides the user-facing interface through a browser-based dashboard built with modern web technologies.

**Component Architecture**

.. code-block:: text

    ┌─────────────────────────────────────────────────┐
    │              ecoKI Dashboard                    │
    │           (Browser Interface)                   │
    ├─────────────────────────────────────────────────┤
    │  ┌─────────────────┐  ┌─────────────────────┐  │
    │  │   Interactive    │  │     Pipeline        │  │
    │  │ Visualizations   │  │   Configurator      │  │
    │  │ (Charts/Graphs)  │  │  (Drag-and-drop)    │  │
    │  └─────────────────┘  └─────────────────────┘  │
    │  ┌─────────────────────────────────────────┐   │
    │  │           Control Panel                  │   │
    │  │            (Run/Stop)                    │   │
    │  └─────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────┘
                         │
                         ▼
                  REST API (JSON)

The frontend comprises three primary components:

**1. Interactive Visualizations**

Real-time charts and graphs displaying:

- Live sensor data streams from industrial equipment
- Model predictions and confidence intervals
- Energy consumption metrics and optimization targets
- Historical trend analysis

**2. Pipeline Configurator**

A visual, drag-and-drop interface for constructing ML pipelines:

- Users connect Building Blocks as nodes in a directed acyclic graph (DAG)
- Visual validation of port compatibility (type checking at design time)
- Configuration panels for each block's hyperparameters
- Pipeline versioning and template management

**3. Control Panel**

Operational controls for pipeline execution:

- Start/Stop/Pause pipeline execution
- Real-time execution status and progress indicators
- Error notifications and diagnostic information

**Communication Protocol**

All frontend-backend communication uses RESTful APIs with JSON payloads. This decision was driven by:

- Universal browser support without plugins
- Easy debugging and inspection of requests
- Compatibility with standard monitoring tools

.. warning::

   **Design Constraint**: The frontend **never** communicates directly with the Resource Pool or Data Layer. All requests are proxied through the Backend's API shell. This ensures consistent authentication, authorization, and audit logging.

Layer 2: Backend — The API Gateway & Orchestration Layer
========================================================

The backend layer, implemented in **Python with FastAPI**, serves as the system's nerve center, handling all request routing, authentication, and pipeline orchestration.

**FastAPI REST Interface**

.. code-block:: python

    # Simplified example of the API structure
    from fastapi import FastAPI, Depends, HTTPException
    from pydantic import BaseModel
    
    app = FastAPI(
        title="EcoKI Backend API",
        description="Low-code ML Platform API Gateway",
        version="1.0.0"
    )
    
    class PipelineConfig(BaseModel):
        """
        Pipeline configuration schema.
        Defines the DAG structure and block parameters.
        """
        pipeline_id: str
        blocks: list[BlockConfig]
        connections: list[Connection]
        metadata: dict
    
    @app.post("/api/v1/pipelines/execute")
    async def execute_pipeline(
        config: PipelineConfig,
        user: User = Depends(get_current_user)
    ):
        """
        Execute a pipeline with the given configuration.
        
        This endpoint:
        1. Validates the pipeline configuration
        2. Resolves block dependencies
        3. Delegates to PipelineThreadManager for execution
        4. Returns execution handle for status polling
        """
        # Validation and execution logic
        ...

The API layer acts as a **Gatekeeper**, performing:

- **Authentication & Authorization**: JWT-based token validation
- **Request Validation**: Pydantic schemas ensure type safety
- **Rate Limiting**: Protecting backend resources from abuse
- **Request Logging**: Audit trail for compliance

**Pipeline Manager: The Orchestration Core**

The ``PipelineThreadManager`` is the heart of the backend, responsible for:

.. code-block:: python

    class PipelineThreadManager:
        """
        Orchestrates pipeline execution with configurable concurrency.
        
        Responsibilities:
        - Parse pipeline DAG and determine execution order
        - Manage thread pool for parallel block execution
        - Handle inter-block data transfer
        - Coordinate with Execution Engine via message passing
        """
        
        def __init__(self, max_workers: int = 4):
            # Thread pool for concurrent block execution
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
            
            # Pipeline execution state (externalized)
            self.execution_states: dict[str, PipelineState] = {}
        
        def schedule_pipeline(self, config: PipelineConfig) -> str:
            """
            Schedule a pipeline for execution.
            
            Algorithm:
            1. Topological sort of DAG to determine execution order
            2. Identify parallelizable blocks (no dependencies)
            3. Submit to thread pool with dependency callbacks
            
            Returns: execution_id for status tracking
            """
            # Topological sort for execution ordering
            execution_order = self._topological_sort(config)
            
            # Submit blocks respecting dependencies
            ...
        
        def _topological_sort(self, config: PipelineConfig) -> list[str]:
            """
            Kahn's algorithm for DAG linearization.
            
            Ensures blocks execute only after their dependencies complete.
            """
            # Implementation of topological sort
            ...

**OpenAPI/Swagger Documentation**

FastAPI's automatic OpenAPI schema generation provides:

- Interactive API documentation at ``/docs``
- Client SDK generation for multiple languages
- Contract-first development workflow

Layer 3: Execution Engine — The Computational Core
==================================================

The Execution Engine is where the actual ML computation happens. It's designed around two key abstractions: **Pipelines** and **Executors**.

**Pipeline as a Directed Acyclic Graph (DAG)**

A Pipeline is a sequence of connected Building Blocks forming a DAG:

.. code-block:: text

    ┌────────────┐     ┌────────────┐     ┌────────────┐
    │   Data     │────▶│   Pre-     │────▶│   Model    │
    │   Reader   │     │ processing │     │  Training  │
    └────────────┘     └────────────┘     └────────────┘
          │                                      │
          │           ┌────────────┐            │
          └──────────▶│  Feature   │────────────┘
                      │ Selection  │
                      └────────────┘

**DAG Properties Enforced**:

- **Acyclicity**: Prevents infinite loops in execution
- **Single Source/Sink**: Clear entry and exit points
- **Type Compatibility**: Output ports must match input port types

**The Executor Pattern**

Executors provide a clean separation between computation logic and lifecycle management:

.. code-block:: python

    from abc import ABC, abstractmethod
    from typing import Generic, TypeVar
    
    T = TypeVar('T', bound='BuildingBlock')
    
    class Executor(ABC, Generic[T]):
        """
        Abstract base class for all executors.
        
        Executors manage the lifecycle of Building Block execution,
        handling resource allocation, error recovery, and telemetry.
        """
        
        @abstractmethod
        def execute(self, block: T, inputs: dict) -> dict:
            """
            Execute a building block with given inputs.
            
            Args:
                block: The Building Block instance to execute
                inputs: Dictionary mapping port names to input data
                
            Returns:
                Dictionary mapping output port names to result data
            """
            pass
        
        @abstractmethod
        def validate(self, block: T, inputs: dict) -> bool:
            """
            Validate inputs before execution.
            
            Performs type checking and constraint validation.
            """
            pass


    class PipelineExecutor(Executor):
        """
        Executor for entire pipelines.
        
        Handles DAG traversal, parallel execution scheduling,
        and cross-block data transfer.
        """
        
        def execute(self, pipeline: Pipeline, inputs: dict) -> dict:
            """
            Execute a complete pipeline.
            
            Algorithm:
            1. Initialize execution context
            2. Execute blocks in topological order
            3. Manage intermediate results
            4. Return final outputs
            """
            # Create execution context for this run
            context = ExecutionContext(pipeline.id)
            
            # Execute each block in order
            for block_id in self._execution_order(pipeline):
                block = pipeline.blocks[block_id]
                block_inputs = self._resolve_inputs(block, context)
                
                # Delegate to BuildingBlockExecutor
                executor = BuildingBlockExecutor()
                result = executor.execute(block, block_inputs)
                
                # Store results in context for downstream blocks
                context.store_result(block_id, result)
            
            return context.get_final_outputs()


    class BuildingBlockExecutor(Executor):
        """
        Executor for individual Building Blocks.
        
        Provides isolation, error handling, and resource management
        for single block execution.
        """
        
        def execute(self, block: BuildingBlock, inputs: dict) -> dict:
            """
            Execute a single Building Block.
            
            Wraps the block's run() method with:
            - Input validation
            - Exception handling
            - Performance telemetry
            - Resource cleanup
            """
            # Validate inputs match expected schema
            if not self.validate(block, inputs):
                raise ValidationError(f"Invalid inputs for {block.name}")
            
            try:
                # Record execution start time
                start_time = time.perf_counter()
                
                # Execute the stateless block
                result = block.run(**inputs)
                
                # Record telemetry
                duration = time.perf_counter() - start_time
                self._record_metrics(block, duration)
                
                return result
                
            except Exception as e:
                # Log error with full context
                self._log_error(block, inputs, e)
                raise ExecutionError(f"Block {block.name} failed: {e}")

**Building Block: The Atomic Unit**

Building Blocks are the fundamental computational units:

.. code-block:: python

    from abc import ABC, abstractmethod
    from dataclasses import dataclass
    from typing import Any
    
    @dataclass
    class Port:
        """
        Defines an input or output port for a Building Block.
        
        Ports enable type-safe connections between blocks.
        """
        name: str
        dtype: type
        description: str
        required: bool = True
    
    
    class BuildingBlock(ABC):
        """
        Abstract base class for all Building Blocks.
        
        Key Design Principles:
        - STATELESS: No internal state between run() calls
        - SINGLE RESPONSIBILITY: One well-defined operation
        - TYPE-SAFE PORTS: Explicit input/output contracts
        
        Example blocks:
        - DataReader: Read from MongoDB, CSV, etc.
        - DataImputer: Handle missing values
        - XGBoostRegressor: Train XGBoost model
        - FeatureSelector: Recursive feature elimination
        """
        
        @property
        @abstractmethod
        def input_ports(self) -> list[Port]:
            """Define expected inputs with types."""
            pass
        
        @property
        @abstractmethod
        def output_ports(self) -> list[Port]:
            """Define outputs with types."""
            pass
        
        @abstractmethod
        def run(self, **inputs) -> dict[str, Any]:
            """
            Execute the block's computation.
            
            MUST be stateless: same inputs always produce same outputs.
            """
            pass


    class DataImputer(BuildingBlock):
        """
        Handles missing values in tabular data.
        
        Strategies:
        - mean: Replace with column mean
        - median: Replace with column median
        - mode: Replace with most frequent value
        - constant: Replace with specified value
        """
        
        def __init__(self, strategy: str = "mean", fill_value: float = None):
            self.strategy = strategy
            self.fill_value = fill_value
        
        @property
        def input_ports(self) -> list[Port]:
            return [
                Port("data", pd.DataFrame, "Input dataframe with missing values"),
                Port("columns", list, "Columns to impute", required=False)
            ]
        
        @property
        def output_ports(self) -> list[Port]:
            return [
                Port("data", pd.DataFrame, "Dataframe with imputed values"),
                Port("statistics", dict, "Imputation statistics")
            ]
        
        def run(self, data: pd.DataFrame, columns: list = None) -> dict:
            """
            Impute missing values in the dataframe.
            
            Returns the imputed data and statistics about what was filled.
            """
            # Implementation of imputation logic
            ...

.. tip::

   **Why this pattern?** The Building Block abstraction enables:
   
   - **Composability**: Blocks can be freely combined into pipelines
   - **Testability**: Each block can be unit tested in isolation
   - **Reusability**: Same block serves multiple use cases
   - **Discoverability**: Ports document the block's contract

Layer 4: Resource Pool — Configuration & Static Assets
======================================================

The Resource Pool is a **standalone microservice** responsible for serving configuration files, metadata, and static content.

**Architecture**

.. code-block:: text

    ┌─────────────────────────────────────────────┐
    │            Resource Pool Service            │
    ├─────────────────────────────────────────────┤
    │  ┌─────────────────────────────────────┐   │
    │  │       Resource Pool Router          │   │
    │  │          (API Router)               │   │
    │  └─────────────────────────────────────┘   │
    │                    │                        │
    │  ┌─────────┬───────┴───────┬───────────┐  │
    │  │ Config  │   Metadata    │  Static   │  │
    │  │  Files  │   (Block      │  Content  │  │
    │  │ (JSON/  │ descriptions) │ (Success  │  │
    │  │  YAML)  │               │  Stories) │  │
    │  └─────────┴───────────────┴───────────┘  │
    └─────────────────────────────────────────────┘

**Storage Containers**

The Resource Pool manages three types of artifacts:

**1. Config Files (JSON/YAML)**

Pipeline configurations, system settings, and deployment parameters:

.. code-block:: yaml

    # Example: settings.yaml
    pipeline:
      max_concurrent_blocks: 4
      timeout_seconds: 3600
      retry_policy:
        max_retries: 3
        backoff_multiplier: 2.0
    
    database:
      connection_string: "${MONGO_URI}"
      pool_size: 10
    
    logging:
      level: INFO
      format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

**2. Metadata (Building Block Descriptions)**

Registry of available blocks with their documentation:

.. code-block:: json

    {
      "blocks": [
        {
          "id": "data_imputer_v1",
          "name": "DataImputer",
          "version": "1.0.0",
          "category": "preprocessing",
          "description": "Handle missing values with configurable strategies",
          "input_ports": [
            {"name": "data", "type": "DataFrame", "required": true}
          ],
          "output_ports": [
            {"name": "data", "type": "DataFrame"}
          ],
          "parameters": [
            {"name": "strategy", "type": "enum", "values": ["mean", "median", "mode"]}
          ]
        }
      ]
    }

**3. Static Content**

Non-code assets including:

- Success stories and case studies
- Energy efficiency scenario templates
- Documentation and help content

**Why a Standalone Service?**

Separating the Resource Pool as an independent service provides:

- **Independent Scaling**: Config serving can scale independently of computation
- **Caching**: Static content can be aggressively cached at the edge
- **Deployment Flexibility**: Config changes don't require backend redeployment
- **Version Management**: Configuration versioning separate from code

Layer 5: Data Acquisition & Storage — Industrial IoT Integration
================================================================

The bottom layer handles the challenging task of ingesting data from diverse industrial sources.

**Data Flow Architecture**

.. code-block:: text

    ┌───────────┐     ┌────────────────────┐     ┌──────────────┐
    │ Plant/    │────▶│ Plant Communication│────▶│   MongoDB    │
    │ Factory   │     │     Adapter        │     │ (On-Premise/ │
    │ Sensors   │     │ (Bridge/Ingestion) │     │ Off-Premise) │
    └───────────┘     └────────────────────┘     └──────────────┘
         │                     │                        │
         │              Normalized Data                 │
         │                     │                        │
    Raw Data              Protocol                  Flexible
    (Various              Translation               Schema
    Protocols)                                      Storage

**Plant Communication Adapter**

The adapter layer bridges the gap between industrial protocols and our platform:

.. code-block:: python

    class PlantCommunicationAdapter:
        """
        Bridge between industrial sensors and the EcoKI platform.
        
        Responsibilities:
        - Protocol translation (OPC-UA, MQTT, Modbus → internal format)
        - Data normalization and validation
        - Buffering for network resilience
        - Timestamp synchronization
        """
        
        def __init__(self, config: AdapterConfig):
            self.config = config
            self.buffer = CircularBuffer(max_size=config.buffer_size)
            self.normalizers = self._load_normalizers()
        
        def ingest(self, raw_data: bytes, protocol: str) -> NormalizedData:
            """
            Ingest raw sensor data and normalize for storage.
            
            Steps:
            1. Parse protocol-specific format
            2. Validate data integrity
            3. Normalize to standard schema
            4. Apply unit conversions
            5. Add metadata (source, timestamp, quality)
            """
            # Protocol-specific parsing
            parser = self._get_parser(protocol)
            parsed = parser.parse(raw_data)
            
            # Normalize to standard format
            normalized = self._normalize(parsed)
            
            # Buffer for batch writing
            self.buffer.append(normalized)
            
            return normalized

**Why MongoDB?**

The choice of MongoDB as the primary data store was driven by:

- **Schema Flexibility**: Industrial data varies wildly between plants
- **Time-Series Optimization**: Native support for time-series collections
- **Horizontal Scaling**: Sharding for high-volume sensor data
- **On-Premise Deployment**: Critical for data sovereignty requirements

.. code-block:: python

    # MongoDB time-series collection for sensor data
    db.create_collection(
        "sensor_readings",
        timeseries={
            "timeField": "timestamp",
            "metaField": "sensor_metadata",
            "granularity": "seconds"
        }
    )

**Dual Deployment Model**

The platform supports both on-premise and cloud deployment:

- **On-Premise**: For sensitive industrial data that cannot leave the factory
- **Off-Premise**: For aggregated analytics and cross-site comparisons

Cross-Cutting Concerns
======================

Several concerns span all layers and require dedicated attention.

**Containerization with Docker & Kubernetes**

Each layer is packaged as a Docker container for consistent deployment:

.. code-block:: dockerfile

    # Example Dockerfile for the Backend service
    FROM python:3.11-slim
    
    WORKDIR /app
    
    # Install Poetry for dependency management
    RUN pip install poetry
    
    # Copy dependency files
    COPY pyproject.toml poetry.lock ./
    
    # Install dependencies (no dev dependencies in production)
    RUN poetry config virtualenvs.create false \
        && poetry install --no-dev --no-interaction
    
    # Copy application code
    COPY src/ ./src/
    
    # Run with Uvicorn ASGI server
    CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]

Kubernetes orchestrates the containers with:

- **Horizontal Pod Autoscaling**: Scale based on CPU/memory usage
- **Service Discovery**: Internal DNS for inter-service communication
- **ConfigMaps/Secrets**: Environment-specific configuration
- **Rolling Deployments**: Zero-downtime updates

**CI/CD Pipeline with GitLab**

The development workflow is fully automated:

.. code-block:: yaml

    # .gitlab-ci.yml
    stages:
      - lint
      - test
      - build
      - deploy
    
    lint:
      stage: lint
      script:
        - poetry run black --check src/
        - poetry run mypy src/
        - poetry run ruff src/
    
    test:
      stage: test
      script:
        - poetry run pytest tests/ --cov=src --cov-report=xml
      coverage: '/TOTAL.*\s+(\d+%)/'
    
    build:
      stage: build
      script:
        - docker build -t ecoki-backend:${CI_COMMIT_SHA} .
        - docker push registry/ecoki-backend:${CI_COMMIT_SHA}
    
    deploy:
      stage: deploy
      script:
        - kubectl set image deployment/backend backend=registry/ecoki-backend:${CI_COMMIT_SHA}
      only:
        - main

Design Trade-offs & Lessons Learned
===================================

Every architecture involves trade-offs. Here are the key decisions we made and their implications.

**Trade-off 1: Stateless Blocks vs. Execution Overhead**

*Decision*: All Building Blocks are stateless.

*Benefit*: Simplified testing, debugging, and horizontal scaling.

*Cost*: Additional overhead for state serialization between blocks. Large intermediate results (e.g., DataFrames) must be serialized and deserialized.

*Mitigation*: Implemented in-memory data passing within the same executor process, only serializing for cross-process or persistent storage.

**Trade-off 2: REST API vs. WebSocket for Real-Time Updates**

*Decision*: Primary communication via REST, with polling for status updates.

*Benefit*: Simpler implementation, better tooling support, stateless backend.

*Cost*: Higher latency for real-time updates, increased network traffic from polling.

*Future*: WebSocket upgrade planned for real-time streaming use cases.

**Trade-off 3: Monorepo vs. Polyrepo**

*Decision*: Single monorepo for all platform components.

*Benefit*: Atomic changes across layers, shared tooling, simplified CI/CD.

*Cost*: Larger repository size, potential for coupling between services.

*Mitigation*: Strict directory structure and code review policies enforce boundaries.

**Trade-off 4: Python Throughout vs. Polyglot Architecture**

*Decision*: Python for all components (frontend uses Python-based Panel library).

*Benefit*: Single language expertise required, easy ML library integration (scikit-learn, PyTorch, XGBoost).

*Cost*: Performance limitations for CPU-intensive frontend operations.

*Mitigation*: Critical paths optimized with Cython/Numba; frontend offloads computation to backend.

Conclusion
==========

The EcoKI architecture demonstrates how thoughtful system design can enable complex ML capabilities while maintaining accessibility for non-expert users. The key takeaways are:

1. **Layer Separation**: Clear boundaries between presentation, orchestration, execution, configuration, and data layers enable independent evolution.

2. **Stateless Computation**: The Building Block pattern, combined with the Executor pattern, provides a powerful abstraction for composable ML workflows.

3. **Protocol Standardization**: REST APIs with JSON payloads and well-defined schemas create clear contracts between components.

4. **Industrial Pragmatism**: The architecture accommodates real-world constraints like on-premise deployment, diverse industrial protocols, and data sovereignty requirements.

Building a production ML platform is as much about software engineering as it is about machine learning. The abstractions you choose early, the patterns you establish, and the trade-offs you make consciously will determine your system's long-term viability.

.. epigraph::

   Good architecture is not about predicting the future; it's about creating flexibility to adapt when the future arrives.
   
   -- *A lesson learned from EcoKI*

References & Further Reading
============================

.. [1] Kleppmann, M. (2017). *Designing Data-Intensive Applications*. O'Reilly Media.

.. [2] Newman, S. (2021). *Building Microservices*, 2nd Edition. O'Reilly Media.

.. [3] Fowler, M. (2002). *Patterns of Enterprise Application Architecture*. Addison-Wesley.

.. [4] FastAPI Documentation. https://fastapi.tiangolo.com/

.. [5] MongoDB Time Series Collections. https://www.mongodb.com/docs/manual/core/timeseries-collections/

.. [6] Kubernetes Documentation. https://kubernetes.io/docs/

.. [7] Sculley, D., et al. (2015). Hidden Technical Debt in Machine Learning Systems. *NeurIPS*.


