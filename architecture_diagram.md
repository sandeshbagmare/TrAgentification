# 📊 Multi-Agent AI Trading System Architecture

```mermaid
graph TB
    A[Data Ingestion Agent] --> B[Feature & Signal Agent]
    B --> C[Automated ML Agent]
    C --> D[Strategy Orchestrator Agent]
    A --> C
    A --> D
    B --> D
    D --> E[Risk Management Agent]
    E --> F[Execution Agent]
    F --> G[Upstox API]
    E --> H[Supervisor & Monitoring Agent]
    F --> H
    H --> I[Real-time Dashboard]
    H --> J[Alert System]
    E --> J
    D --> E
    A --> H
    C --> H
    subgraph "Data Layer"
        A
    end
    subgraph "Strategy Layer"
        B
        C
        D
    end
    subgraph "Risk Layer"
        E
    end
    subgraph "Execution Layer"
        F
    end
    subgraph "Monitoring Layer"
        H
        I
        J
    end
end
```