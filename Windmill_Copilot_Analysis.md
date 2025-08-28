# Windmill Copilot Component System: In-Depth Analysis

## Executive Summary

The Windmill copilot system is a sophisticated AI-powered code generation and workflow creation platform that transforms natural language instructions into executable scripts and complex workflows. This document provides a comprehensive analysis of how the system creates prompts and generates flows from completion API responses.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Libraries](#core-libraries)
3. [Prompt System Architecture](#prompt-system-architecture)
4. [Flow Creation Process](#flow-creation-process)
5. [Prompt Engineering Patterns](#prompt-engineering-patterns)
6. [Completion API Integration](#completion-api-integration)
7. [Advanced Features](#advanced-features)
8. [Key Innovations](#key-innovations)
9. [Security and Validation](#security-and-validation)
10. [Technical Implementation Details](#technical-implementation-details)

## Architecture Overview

The Windmill copilot system is built around several interconnected components that work together to provide AI-powered workflow creation:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   UI Components │────│  Chat Manager    │────│  Completion API │
│   (Svelte)      │    │  (State Mgmt)    │    │  (OpenAI/etc)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Prompt Templates│    │   Tool System    │    │ Response Parser │
│ (Multi-language)│    │ (17 Flow Tools)  │    │ (Code Extraction)│
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Core Libraries

### Main Library (`lib.ts`)

The core library provides the foundation for AI interactions:

#### Key Functions

```typescript
// Central orchestration function
async function copilot(
  scriptOptions: CopilotOptions,
  generatedCode: Writable<string>,
  abortController: AbortController,
  generatedExplanation?: Writable<string>
): Promise<string>

// Streaming chat completions
async function getCompletion(
  messages: ChatCompletionMessageParam[],
  abortController: AbortController,
  tools?: OpenAI.Chat.Completions.ChatCompletionTool[]
)

// Single completion requests
async function getNonStreamingCompletion(
  messages: ChatCompletionMessageParam[],
  abortController: AbortController,
  testOptions?: TestOptions
): Promise<string>
```

#### AI Provider Support

The system supports multiple AI providers with automatic model selection:

```typescript
export const AI_DEFAULT_MODELS: Record<AIProvider, string[]> = {
  openai: ['gpt-5', 'gpt-5-mini', 'gpt-4o', 'gpt-4o-mini', 'o4-mini', 'o3'],
  azure_openai: OPENAI_MODELS,
  anthropic: ['claude-sonnet-4-0', 'claude-3-5-haiku-latest'],
  mistral: ['codestral-latest'],
  deepseek: ['deepseek-chat', 'deepseek-reasoner'],
  googleai: ['gemini-2.0-flash', 'gemini-1.5-flash'],
  groq: ['llama-3.3-70b-versatile'],
  openrouter: ['meta-llama/llama-3.2-3b-instruct:free'],
  togetherai: ['meta-llama/Llama-3.3-70B-Instruct-Turbo'],
  customai: []
}
```

#### Context Window Management

```typescript
function getModelContextWindow(model: string): number {
  if (model.startsWith('gpt-4.1') || model.startsWith('gemini')) {
    return 1000000  // 1M tokens
  } else if (model.startsWith('gpt-5')) {
    return 400000   // 400K tokens
  } else if (model.startsWith('claude')) {
    return 200000   // 200K tokens
  }
  // ... additional model mappings
}
```

## Prompt System Architecture

### Prompt Configuration Structure

The system uses a hierarchical prompt configuration:

```typescript
interface PromptsConfig {
  system: string  // System-level instructions for AI behavior
  prompts: {
    [language: string]: {
      prompt: string  // Language-specific user prompt template
    }
  }
}
```

### Three Operation Types

#### 1. Generation Mode (GEN_CONFIG)
Creates code from scratch based on natural language descriptions.

**System Prompt:**
```
You are a helpful coding assistant for Windmill, a developer platform for running scripts. 
You write code as instructed by the user. Each user message includes some contextual 
information which should guide your answer.

Only output code. Wrap the code in a code block.
Put explanations directly in the code as comments.
```

#### 2. Edit Mode (EDIT_CONFIG)
Modifies existing code based on change instructions.

**System Prompt:**
```
You are a helpful coding assistant for Windmill, a developer platform for running scripts. 
You modify code as instructed by the user. Each user message includes some contextual 
information which should guide your answer.

Only output code. Wrap the code in a code block. 
Put explanations directly in the code as comments.
Return the complete modified code.
```

#### 3. Fix Mode (FIX_CONFIG)
Fixes code errors and provides explanations.

**System Prompt:**
```
You are a helpful coding assistant for Windmill, a developer platform for running scripts. 
You fix the code shared by the user. Each user message includes some contextual information 
which should guide your answer.

Only output code. Wrap the code in a code block. 
Explain the error and the fix after generating the code inside an <explanation> tag.
Also put explanations directly in the code as comments.
Return the complete fixed code.
```

### Language-Specific Prompts

Each supported language has specialized prompts with specific conventions:

#### Python Example
```typescript
"python3": {
  "prompt": `<contextual_information>
You have to write a function in Python called "main". Specify the parameter types. 
Do not call the main function. You should generally return the result.
The "main" function cannot be async. If you need to use async code, you can use the asyncio library.

You can take as parameters resources which are dictionaries containing credentials or 
configuration information. For Windmill to correctly detect the resources to be passed, 
the resource type name has to be exactly as specified in the following list:
<resourceTypes>
{resourceTypes}
</resourceTypes>

You need to define the type of the resources that are needed before the main function, 
but only include them if they are actually needed to achieve the function purpose.
The resource type name has to be exactly as specified (has to be IN LOWERCASE).
</contextual_information>
My instructions: {description}`
}
```

#### TypeScript/Deno Example
```typescript
"deno": {
  "prompt": `<contextual_information>
You have to write TypeScript code and export a "main" function like this: 
"export async function main(...)" and specify the parameter types but do not call it. 
You should generally return the result.

You can import deno libraries or you can also import npm libraries like that: 
"import ... from "npm:{package}";". The fetch standard method is available globally.

You can take as parameters resources which are dictionaries containing credentials or 
configuration information. For Windmill to correctly detect the resources to be passed, 
the resource type name has to be exactly as specified in the following list:
<resourceTypes>
{resourceTypes}
</resourceTypes>
</contextual_information>
My instructions: {description}`
}
```

## Flow Creation Process

### Flow AI Chat System

The flow creation system operates through the `AIChatManager` which manages different operational modes:

```typescript
export enum AIMode {
  SCRIPT = 'script',    // Individual script editing
  FLOW = 'flow',        // Flow creation and modification
  NAVIGATOR = 'navigator', // Application navigation
  API = 'API',          // API interaction
  ASK = 'ask'           // General questions
}
```

### Flow Tools Architecture

The flow creation system provides 17 specialized tools for comprehensive flow manipulation:

#### Structure Management Tools

```typescript
// Add steps at specific locations
const addStepSchema = z.object({
  location: insertLocationSchema,
  step: newStepSchema
})

// Remove steps
const removeStepSchema = z.object({
  id: z.string().describe('The id of the step to remove')
})

// Manage branch logic
const addBranchSchema = z.object({
  id: z.string().describe('The id of the step to add the branch to')
})

const removeBranchSchema = z.object({
  id: z.string().describe('The id of the step to remove the branch from'),
  branchIndex: z.number().describe('The index of the branch to remove, starting at 0')
})
```

#### Content Management Tools

```typescript
// Update step code
const setCodeSchema = z.object({
  id: z.string().describe('The id of the step to set the code for'),
  code: z.string().describe('The code to apply')
})

// Configure step parameters
const setStepInputsSchema = z.object({
  id: z.string().describe('The id of the step to set the inputs for'),
  inputs: z.string().describe('The inputs to set for the step')
})

// Define flow schema
const setFlowInputsSchemaSchema = z.object({
  schema: z.string().describe('JSON string of the flow inputs schema (draft 2020-12)')
})
```

#### Discovery & Testing Tools

```typescript
// Find existing scripts
const searchScriptsSchema = z.object({
  query: z.string().describe('The query to search for, e.g. send email, list stripe invoices, etc..')
})

// Search for integrations
const resourceTypeToolSchema = z.object({
  query: z.string().describe('The query to search for, e.g. stripe, google, etc..'),
  language: langSchema.describe('The programming language the code using the resource type will be written in')
})

// Execute tests
const testRunFlowSchema = z.object({
  args: z.object({}).nullable().optional().describe('Arguments to pass to the flow')
})
```

### Insert Location System

The system supports precise step placement:

```typescript
type InsertLocation = 
  | { type: 'start' }  // Beginning of flow
  | { type: 'after', afterId: string }  // After specific step
  | { type: 'start_inside_forloop', inside: string }  // Inside loop
  | { type: 'start_inside_branch', inside: string, branchIndex: number }  // Inside branch
  | { type: 'preprocessor' }  // Before flow execution
  | { type: 'failure' }  // Error handling
```

### Step Type System

```typescript
type NewStep = 
  | { 
      type: 'rawscript', 
      language: string, 
      summary: string 
    }  // Custom code
  | { 
      type: 'script', 
      path: string 
    }  // Existing script
  | { type: 'forloop' }     // Iteration logic
  | { type: 'branchall' }   // Parallel execution
  | { type: 'branchone' }   // Conditional logic
```

## Prompt Engineering Patterns

### Context Injection System

The system automatically enriches prompts with relevant context:

#### Resource Type Integration

```typescript
export async function addResourceTypes(scriptOptions: CopilotOptions, prompt: string) {
  if (['deno', 'bun', 'nativets', 'python3', 'php'].includes(scriptOptions.language)) {
    const resourceTypes = await getResourceTypes(scriptOptions)
    const resourceTypesText = formatResourceTypes(
      resourceTypes,
      ['deno', 'bun', 'nativets'].includes(scriptOptions.language)
        ? 'typescript'
        : (scriptOptions.language as 'python3' | 'php')
    )
    prompt = prompt.replace('{resourceTypes}', resourceTypesText)
  }
  return prompt
}
```

#### Database Schema Integration

```typescript
function addDBSchema(scriptOptions: CopilotOptions, prompt: string) {
  const { dbSchema, language } = scriptOptions
  if (
    dbSchema &&
    ['postgresql', 'mysql', 'snowflake', 'bigquery', 'mssql', 'graphql', 'oracledb'].includes(language) &&
    language === dbSchema.lang
  ) {
    let { stringified } = dbSchema
    if (dbSchema.lang === 'graphql') {
      prompt = prompt + '\nHere is the GraphQL schema: <schema>\n' + stringified + '\n</schema>'
    } else {
      prompt = prompt +
        "\nHere's the database schema, each column is in the format [name, type, required, default?]: <dbschema>\n" +
        stringified +
        '\n</dbschema>'
    }
  }
  return prompt
}
```

### Variable Replacement System

The system uses a sophisticated variable replacement mechanism:

```typescript
async function getPrompts(scriptOptions: CopilotOptions) {
  const promptsConfig = PROMPTS_CONFIGS[scriptOptions.type]
  let prompt = promptsConfig.prompts[scriptOptions.language].prompt
  
  // Core replacements
  if (scriptOptions.type !== 'fix') {
    prompt = prompt.replace('{description}', scriptOptions.description)
  }
  
  if (scriptOptions.type !== 'gen') {
    prompt = prompt.replace('{code}', scriptOptions.code)
  }
  
  if (scriptOptions.type === 'fix') {
    prompt = prompt.replace('{error}', scriptOptions.error)
  }
  
  // Context enrichment
  prompt = await addResourceTypes(scriptOptions, prompt)
  prompt = addDBSchema(scriptOptions, prompt)
  
  return { prompt, systemPrompt: promptsConfig.system }
}
```

### Flow Context Understanding

For flow operations, the system provides comprehensive context:

```typescript
export function prepareFlowUserMessage(
  instructions: string,
  flowAndSelectedId?: { flow: ExtendedOpenFlow; selectedId: string },
  selectedContext?: ContextElement[]
): ChatCompletionUserMessageParam {
  const flow = flowAndSelectedId?.flow
  const selectedId = flowAndSelectedId?.selectedId
  
  const contextInstructions = selectedContext ? buildContextString(selectedContext) : ''
  
  if (!flow || !selectedId) {
    return {
      role: 'user',
      content: `## INSTRUCTIONS:\n${instructions}`
    }
  }
  
  const codePieces = selectedContext?.filter((c) => c.type === 'code_piece') ?? []
  const flowModulesYaml = applyCodePiecesToFlowModules(codePieces, flow.value.modules)
  
  let flowContent = `## FLOW:
flow_input schema:
${JSON.stringify(flow.schema ?? emptySchema())}

flow modules:
${flowModulesYaml}

preprocessor module:
${YAML.stringify(flow.value.preprocessor_module)}

failure module:
${YAML.stringify(flow.value.failure_module)}

currently selected step:
${selectedId}`
  
  flowContent += contextInstructions
  flowContent += `\n\n## INSTRUCTIONS:\n${instructions}`
  
  return {
    role: 'user',
    content: flowContent
  }
}
```

## Completion API Integration

### Streaming Response Processing

The system processes streaming responses in real-time:

```typescript
export async function copilot(
  scriptOptions: CopilotOptions,
  generatedCode: Writable<string>,
  abortController: AbortController,
  generatedExplanation?: Writable<string>
) {
  const { prompt, systemPrompt } = await getPrompts(scriptOptions)
  
  const completion = await getCompletion(
    [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: prompt }
    ],
    abortController
  )
  
  let response = ''
  let code = ''
  
  for await (const part of completion) {
    response += getResponseFromEvent(part)
    
    // Extract complete code blocks
    let match = response.match(/```[a-zA-Z]+\n([\s\S]*?)\n```/)
    if (match) {
      code = match[1]
      generatedCode.set(code)
      
      // Handle explanations in fix mode
      if (scriptOptions.type === 'fix') {
        let explanationMatch = response.match(/<explanation>([\s\S]+)<\/explanation>/)
        if (explanationMatch) {
          const explanation = explanationMatch[1].trim()
          generatedExplanation?.set(explanation)
          break
        }
      } else {
        break
      }
    }
    
    // Handle partial code blocks
    match = response.match(/```[a-zA-Z]+\n([\s\S]*)/)
    if (match && !match[1].endsWith('`')) {
      code = match[1]
      generatedCode.set(code)
    }
  }
  
  generatedCode.set(code)
  
  if (code.length === 0) {
    throw new Error('No code block found')
  }
  
  return code
}
```

### Tool Call Processing

The system processes AI tool calls to modify flows:

```typescript
export async function processToolCall<T>({
  tools,
  toolCall,
  helpers,
  toolCallbacks
}: {
  tools: Tool<T>[]
  toolCall: ChatCompletionMessageToolCall
  helpers: T
  toolCallbacks: ToolCallbacks
}): Promise<ChatCompletionMessageParam> {
  try {
    const args = JSON.parse(toolCall.function.arguments || '{}')
    const tool = tools.find((t) => t.def.function.name === toolCall.function.name)
    
    // Check if tool requires confirmation
    const needsConfirmation = tool?.requiresConfirmation
    
    // Add the tool to the display with appropriate status
    toolCallbacks.setToolStatus(toolCall.id, {
      ...(tool?.requiresConfirmation
        ? { content: tool.confirmationMessage ?? 'Waiting for confirmation...' }
        : {}),
      parameters: args,
      isLoading: true,
      needsConfirmation: needsConfirmation,
      showDetails: tool?.showDetails
    })
    
    // If confirmation is needed and we have the callback, wait for it
    if (needsConfirmation && toolCallbacks.requestConfirmation) {
      const confirmed = await toolCallbacks.requestConfirmation(toolCall.id)
      
      if (!confirmed) {
        toolCallbacks.setToolStatus(toolCall.id, {
          content: 'Cancelled by user',
          isLoading: false,
          error: 'Tool execution was cancelled by user',
          needsConfirmation: false
        })
        return {
          role: 'tool' as const,
          tool_call_id: toolCall.id,
          content: 'Tool execution was cancelled by user'
        }
      }
    }
    
    let result = ''
    try {
      result = await callTool({
        tools,
        functionName: toolCall.function.name,
        args,
        workspace: get(workspaceStore) ?? '',
        helpers,
        toolCallbacks,
        toolId: toolCall.id
      })
      toolCallbacks.setToolStatus(toolCall.id, {
        isLoading: false
      })
    } catch (err) {
      console.error(err)
      toolCallbacks.setToolStatus(toolCall.id, {
        isLoading: false,
        error: 'An error occurred while calling the tool'
      })
      const errorMessage = typeof err === 'string' ? err : 'An error occurred while calling the tool'
      result = `Error while calling tool: ${errorMessage}`
    }
    
    return {
      role: 'tool' as const,
      tool_call_id: toolCall.id,
      content: result
    }
  } catch (err) {
    console.error(err)
    return {
      role: 'tool' as const,
      tool_call_id: toolCall.id,
      content: 'Error while calling tool'
    }
  }
}
```

## Advanced Features

### Visual Change Management

The system provides sophisticated diff visualization:

```typescript
export type VisualChange =
  | {
      type: 'added_inline'
      position: { line: number; column: number }
      value: string
      options?: { greenHighlight?: boolean }
    }
  | {
      type: 'added_block'
      position: { afterLineNumber: number }
      value: string
      options?: {
        greenHighlight?: boolean
        review?: {
          acceptFn: () => void
          rejectFn: () => void
        }
        extraChanges?: VisualChange[]
      }
    }
  | {
      type: 'deleted'
      range: {
        startLine: number
        startColumn: number
        endLine: number
        endColumn: number
      }
      options?: {
        isWholeLine?: boolean
        review?: {
          acceptFn: () => void
          rejectFn: () => void
        }
      }
    }
```

### Context-Aware Input Generation

The step input generation system analyzes flow structure:

```typescript
async function generateStepInputs() {
  const flow: Flow = JSON.parse(JSON.stringify(flowStore.val))
  const idOrders = dfs(flow.value.modules, (x) => x.id)
  const upToIndex = idOrders.indexOf($selectedId)
  
  const flowDetails =
    'Take into account the following information for never tested results:\n<flowDetails>\n' +
    yamlStringifyExceptKeys(sliceModules(flow.value.modules, upToIndex, idOrders), ['lock']) +
    '</flowDetails>'
  
  const availableData = {
    results: pickableProperties?.priorIds,
    flow_input: pickableProperties?.flow_input
  }
  
  const isInsideLoop = availableData.flow_input && 'iter' in availableData.flow_input
  
  const user = `I'm building a workflow which is a DAG of script steps.
The current step is ${$selectedId}, you can find the details for the step and previous ones below:
${flowDetails}

Determine for all the inputs "${argNames.join('", "')}", what to pass either from the previous results of the flow inputs.
All possibles inputs either start with results. or flow_input. and are followed by the key of the input.
${
  isInsideLoop
    ? 'As the step is in a loop, the iterator value is accessible as flow_input.iter.value.'
    : 'As the step is not in a loop, flow_input.iter.value is not available.'
}

Here's a summary of the available data:
<available>
${YAML.stringify(availableData)}</available>

If none of the available results are appropriate, are already used or are more appropriate for other inputs, you can also imagine new flow_input properties which we will create programmatically based on what you provide.

Reply with the most probable answer, do not explain or discuss.
Use javascript object dot notation to access the properties.

Your answer has to be in the following format (one line per input):
input_name1: expression1
input_name2: expression2
...`
  
  generatedContent = await getNonStreamingCompletion(
    [{ role: 'user', content: user }],
    abortController
  )
  
  // Parse and apply generated expressions
  parsedInputs = generatedContent.split('\n').map((x) => x.split(': '))
  
  const exprs = {}
  newFlowInputs = []
  for (const [key, value] of parsedInputs) {
    if (argNames.includes(key)) {
      exprs[key] = value
      if (
        pickableProperties &&
        value.startsWith('flow_input.') &&
        value.split('.')[1] &&
        !(value.split('.')[1] in pickableProperties.flow_input)
      ) {
        newFlowInputs.push(value.split('.')[1])
      }
    }
  }
  generatedExprs?.set(exprs)
}
```

### Metadata Generation

The system automatically generates various types of metadata:

```typescript
const promptConfigs: {
  summary: PromptConfig
  description: PromptConfig
  flowSummary: PromptConfig
  flowDescription: PromptConfig
  agentToolFunctionName: PromptConfig
} = {
  summary: {
    system: `
You are a helpful AI assistant. You generate very brief summaries from scripts.
The summaries need to be as short as possible (maximum 8 words) and only give a global idea. 
Do not specify the programming language. Do not use any punctation. 
Avoid using prepositions and articles.
Examples: List the commits of a GitHub repository, Divide a number by 16, etc..
`,
    user: `
Generate a very short summary for the script below:
'''code
{code}
\`\`\`
`,
    placeholderName: 'code'
  },
  description: {
    system: `
You are a helpful AI assistant. You generate descriptions from scripts.
These descriptions are used to explain to other users what the script does and how to use it.
Be as short as possible to give a global idea, maximum 3-4 sentences.
All scripts export an asynchronous function called main, do not include it in the description.
Do not describe how to call it either.
`,
    user: `
Generate a description for the script below:
'''code
{code}
\`\`\`
`,
    placeholderName: 'code'
  }
  // ... additional configurations
}
```

## Key Innovations

### 1. Flow Context Understanding

The system excels at understanding complete flow context:

- **Dependency Analysis**: Tracks data flow between steps
- **Loop Context**: Understands iterator variables and scope
- **Branch Logic**: Handles conditional execution paths
- **Resource Management**: Maps available resources to step inputs

### 2. Intelligent Variable Mapping

The AI generates appropriate JavaScript expressions:

```javascript
// Step results
results.step1.output_field
results.fetchData.users

// Flow inputs
flow_input.database_url
flow_input.api_key

// Loop iterations
flow_input.iter.value
flow_input.iter.index
```

### 3. Multi-Modal AI Chat

Seamless switching between different operational modes:

- **Script Mode**: Individual script editing with context
- **Flow Mode**: Complete workflow creation and modification
- **Navigator Mode**: Application navigation and data fetching
- **API Mode**: API interaction and testing
- **Ask Mode**: General questions and help

### 4. Visual Diff System

Sophisticated change visualization and management:

- **Inline Changes**: Show additions directly in code
- **Block Changes**: Display new code blocks with review options
- **Deletion Tracking**: Highlight removed code with acceptance/rejection
- **Review Workflow**: User-controlled acceptance of AI changes

## Security and Validation

### Schema Validation

All tool parameters are validated using Zod schemas:

```typescript
const addStepSchema = z.object({
  location: insertLocationSchema,
  step: newStepSchema
})

const setCodeSchema = z.object({
  id: z.string().describe('The id of the step to set the code for'),
  code: z.string().describe('The code to apply')
})
```

### Token Management

Automatic context window management:

```typescript
checkTokenUsageOverLimit = (messages: ChatCompletionMessageParam[]) => {
  const estimatedTokens = messages.reduce((acc, message) => {
    const tokenPerCharacter = 4
    if (message.content) {
      acc += message.content.length / tokenPerCharacter
    }
    if (message.role === 'assistant' && message.tool_calls) {
      acc += JSON.stringify(message.tool_calls).length / tokenPerCharacter
    }
    return acc
  }, 0)
  
  const modelContextWindow = getModelContextWindow(get(copilotSessionModel)?.model ?? '')
  return (
    estimatedTokens >
    modelContextWindow -
      Math.max(modelContextWindow * MAX_TOKENS_THRESHOLD_PERCENTAGE, MAX_TOKENS_HARD_LIMIT)
  )
}
```

### Confirmation System

User approval required for destructive operations:

```typescript
interface Tool<T> {
  def: ChatCompletionTool
  fn: (p: ToolParams) => Promise<string>
  preAction?: (p: { toolCallbacks: ToolCallbacks; toolId: string }) => void
  setSchema?: (helpers: any) => Promise<void>
  requiresConfirmation?: boolean
  confirmationMessage?: string
  showDetails?: boolean
}
```

### Abort Controllers

All operations can be cancelled:

```typescript
async function copilot(
  scriptOptions: CopilotOptions,
  generatedCode: Writable<string>,
  abortController: AbortController,
  generatedExplanation?: Writable<string>
) {
  // Uses abortController.signal throughout async operations
  const completion = await getCompletion(messages, abortController)
  // ...
}
```

## Technical Implementation Details

### File Structure

```
frontend/src/lib/components/copilot/
├── lib.ts                     # Core AI functionality
├── shared.ts                  # Visual changes and diff management
├── utils.ts                   # Helper functions and type compilation
├── flow.ts                    # Flow-specific context types
├── prompts/
│   ├── index.ts              # Prompt configuration exports
│   ├── genPrompt.ts          # Generation prompts
│   ├── editPrompt.ts         # Edit prompts
│   └── fixPrompt.ts          # Fix prompts
├── chat/
│   ├── AIChatManager.svelte.ts    # Central chat management
│   ├── ContextManager.svelte.ts   # Context handling
│   ├── shared.ts                  # Shared chat functionality
│   ├── flow/
│   │   ├── core.ts               # Flow tools and system messages
│   │   ├── FlowAIChat.svelte     # Flow chat UI
│   │   └── utils.ts              # Flow utilities
│   ├── script/
│   │   └── core.ts               # Script-specific tools
│   ├── navigator/
│   │   └── core.ts               # Navigation tools
│   └── api/
│       └── core.ts               # API tools
├── autocomplete/
│   ├── Autocompletor.ts          # Code completion logic
│   └── request.ts                # Completion requests
└── [Component Files]
    ├── ScriptGen.svelte          # Main script generation UI
    ├── MetadataGen.svelte        # Metadata generation
    ├── StepInputsGen.svelte      # Step input generation
    ├── StepInputGen.svelte       # Individual input generation
    └── [Other UI Components]
```

### Component Interaction Flow

```
User Input (Natural Language)
    ↓
AIChatManager (Mode Selection)
    ↓
ContextManager (Context Gathering)
    ↓
Prompt Assembly (Template + Context)
    ↓
AI Provider (Completion API)
    ↓
Response Processing (Code/Tool Extraction)
    ↓
Tool Execution (Flow Modification)
    ↓
Visual Diff (Change Visualization)
    ↓
User Review (Accept/Reject)
    ↓
Final Application (Code/Flow Update)
```

## Conclusion

The Windmill copilot system represents a sophisticated approach to AI-powered workflow development. By combining structured prompting, tool-based execution, intelligent context management, and visual change tracking, it enables natural language flow creation while maintaining precision and control.

Key strengths include:

1. **Comprehensive Language Support**: 15+ programming languages with specialized prompts
2. **Advanced Context Management**: Database schemas, resource types, and flow state integration
3. **Powerful Tool System**: 17 specialized tools for precise flow manipulation
4. **Visual Change Management**: Sophisticated diff visualization and review workflows
5. **Multi-Provider AI Support**: Flexibility across 10+ AI providers
6. **Security and Validation**: Comprehensive validation and user confirmation systems

This architecture enables developers to create complex workflows through natural language while maintaining the precision and reliability required for production systems.

---

*Document generated from comprehensive analysis of Windmill copilot component system*
*Analysis Date: 2024*
*Components Analyzed: 40+ files across frontend/src/lib/components/copilot/*