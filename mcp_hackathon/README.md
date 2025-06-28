# mcp_hackathon

# Data Generation Server for ICD Code Finetuning

This MCP server provides tools for finetuning models to improve medical ICD-10 code prediction accuracy.

## Overview

The Data Generation Server is a Model Control Protocol (MCP) server that facilitates:

1. **Model Probing**: Evaluating model performance on ICD-10 code prediction
2. **Synthetic Data Generation**: Creating training data for model finetuning

## Set up the server

```
{
  "mcpServers": {
      "data_gen_server": {
          "command": "<base_dir>/.local/bin/uv",
          "args": [
              "--directory",
              "<base_dir>/mcp_hackathon/data_gen_server",
              "run",
              "data_gen_server.py"
          ]
      }
  }
}
```

To use MCP tools in Cursor, you need to have your MCP server configured and running. Here's how to use them:

## 1. Verify MCP Configuration

First, make sure your MCP configuration is set up correctly in `/Users/john/.cursor/mcp.json`:
Read file: /Users/john/.cursor/mcp.json
Great! You have two MCP servers configured. Here's how to use them:

## 2. How MCP Tools Work in Cursor

MCP tools in Cursor work through **function calling**. When you mention or request something that can be handled by your MCP tools, Cursor will automatically:

1. **Detect** when a task matches available MCP tools
2. **Call** the appropriate tool with the right parameters
3. **Present** the results to you

## 3. Using MCP Tools

### Method 1: Direct Requests
You can directly ask me to use the tools by mentioning what you want to do:

**Examples:**
- "List all available data generation templates"
- "Generate city information for San Francisco, New York, and Los Angeles"
- "Run the starfish template with this data: [your data]"
- "Generate synthetic data for ICD codes"

### Method 2: Specific Tool References
You can reference tools by name if you know them:

Let me check what tools are available from your MCP servers:
Ran tool
Ran tool
Perfect! Your MCP tools are working. Here's how to use them effectively:

## 4. Available Tools and Usage Examples

### From `starfish_gen_template` server:
- **List templates**: "Show me all available data generation templates"
- **Generate city info**: "Generate information for cities: San Francisco, New York"
- **Run templates**: "Run the starfish/generate_by_topic template"

### From `starfish_data_gen_server` server:
- **Generate data**: "Generate 10 datapoints of synthetic data"
- **Probe model**: "Test the model with 5 ICD code datapoints"
- **Finetune model**: "Finetune the model for ICD code classification"

## 5. How to Use MCP Tools in Practice

### Example 1: Generate Data
```
You: "Generate 10 synthetic datapoints for training"
```
I will automatically call the appropriate MCP tool.

### Example 2: List Available Templates
```
You: "What data generation templates are available?"
```

### Example 3: Run Specific Template
```
You: "Run the generate_by_topic template with topics: AI, Machine Learning"
```

## 6. Best Practices

1. **Be specific** about what you want to accomplish
2. **Provide data** when needed (I'll ask if unclear)
3. **Check results** and iterate if needed
4. **Use natural language** - no need for technical syntax

## 7. Restart Cursor (if needed)

If you just updated your MCP configuration, restart Cursor to ensure the tools are loaded:

1. Quit Cursor completely
2. Reopen it
3. The MCP tools should now be available

## Try It Now!

You can test the MCP tools by asking me to:
- "List all available data generation templates"
- "Generate some sample data"
- "Show me what tools are available"

The tools will work seamlessly in our conversation - just tell me what you want to accomplish!