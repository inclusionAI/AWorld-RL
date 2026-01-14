# AWorld Output Module

The Output module is a flexible and extensible system for managing outputs and artifacts in the AWorld framework.

## Key Features

- **Output Channel Management**: Centralized output handling through `OutputChannel`
- **Multiple Output Types Support**: 
  - Message outputs for text-based communication
  - Artifact outputs for files and code
  - Workspace outputs for managing collections of artifacts
- **Rendering System**: Flexible rendering system with different renderers for various output types
- **Workspace Management**: 
  - Version control for artifacts
  - Local storage support
  - Observer pattern for real-time updates
- **Artifact Management**:
  - Support for different artifact types (Code, Message, etc.)
  - Metadata handling
  - Content versioning

## Class Diagram

```mermaid
classDiagram
    direction TB
%% Base Classes


%% Output Related Classes
    namespace Outputs {
        class Output {
            +metadata: Dict
            +parts: List[OutputPart]
        }
        class OutputPart {
            +content: Any
            +metadata: Dict
        }
        class MessageOutput {
            +source: Any
            +reason_generator: Any
            +response_generator: Any
            +reasoning: str
            +response: Any
            +has_reasoning: bool
            +finished: bool
        }
        class Artifact {
            +artifact_id: str
            +artifact_type: ArtifactType
            +content: Any
            +metadata: Dict
        }
        class ToolCallOutput {
            +tool_id: str
            +content: Any
            +metadata: Dict
        }
        class ToolResultOutput {
            +tool_id: str
            +params: Any
            +content: Any
            +metadata: Dict
        }
        class SearchOutput {
            +artifact_id: str
            +artifact_type: ArtifactType
            +content: Any
            +metadata: Dict
        }
    }

    namespace Output Group {
    %% Panel Related Classes
        class MessagePanel {
            +panel_id: str
            +messages: List[Output]
            +add_output()
            +get_messages_async()
        }

    %% Workspace Related Classes
        class WorkSpace {
            +workspace_id: str
            +name: str
            +artifacts: List[Artifact]
            +observers: List[WorkspaceObserver]
            +create_artifact()
            +update_artifact()
            +delete_artifact()
        }
    }

    namespace Output UI Render {
        class OutputRenderer {
            +render()
        }
        class MessagePanelRenderer {
            +panel: MessagePanel
            +render()
        }
        class WorkspaceRenderer {
            +workspace: WorkSpace
            +render()
        }
    }

    namespace AgentTaskOutput {
    %% Task and Channel Classes
        class Task {
            +task_id: str
            +status: str
            +output_channel: OutputChannel
            +execute()
            +get_result()
        }
        class OutputChannel {
            +chat_id: str
            +outputs: List[Output]
            +message_renderer: MessagePanelRenderer
            +workspace_renderer: WorkspaceRenderer
            +add_output()
            +_dispatch_output()
        }
    }
    Output *-- OutputPart
%% Inheritance Relationships
    Output <|-- MessageOutput
    Output <|-- Artifact
    Output <|-- ToolCallOutput
    Output <|-- ToolResultOutput
    ToolResultOutput <|-- SearchOutput
    OutputRenderer <|-- MessagePanelRenderer
    OutputRenderer <|-- WorkspaceRenderer
%% Composition Relationships
    MessagePanel *-- Output
    MessagePanelRenderer *-- MessagePanel
    WorkspaceRenderer *-- WorkSpace
    WorkSpace *-- Artifact
    OutputChannel *-- MessagePanelRenderer
    OutputChannel *-- WorkspaceRenderer
    Task *-- OutputChannel
```

## OutputChannel Example
- [simple_use.py](../../examples/output/output_channel/simple_use.py)
- [producer_consumer.py](../../examples/output/output_channel/producer_consumer.py)
## Workspace Example
see
- [run.py](../../examples/output/workspace/run.py)