// Day 3 Demo: Building an Agent from Scratch
// Interactive workshop for building agents with pure Python

document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  initBuildPanel();
  initRunPanel();
  initInspectPanel();
});

// =============================================================================
// TAB NAVIGATION
// =============================================================================

function initTabs() {
  const tabBtns = document.querySelectorAll('.tab-btn');
  const panels = document.querySelectorAll('.demo-panel');

  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const tabId = btn.dataset.tab;

      // Update active tab
      tabBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      // Update active panel
      panels.forEach(p => p.classList.remove('active'));
      document.getElementById(`${tabId}-panel`).classList.add('active');
    });
  });
}

// =============================================================================
// PANEL 1: BUILD AGENT
// =============================================================================

const buildSteps = {
  1: {
    code: `from openai import OpenAI

class Agent:
    def __init__(self):
        self.client = OpenAI()
        self.messages = []

    def _call_llm(self):
        completion = self.client.chat.completions.create(
            model="gpt-4o",
            messages=self.messages,
            temperature=0
        )
        return completion.choices[0].message.content`,
    explanation: `<strong>Agent Class:</strong> The foundation. We store an OpenAI client and a message history list. The <code>_call_llm</code> method sends messages to the model and returns the response.`
  },
  2: {
    code: `from openai import OpenAI

class Agent:
    def __init__(self):
        self.client = OpenAI()
        self.messages = []
        self.system = """You are a helpful assistant with tools.

Available tools:
- calculate(expression): Evaluate math expressions
- weather(city): Get weather for a city
- search(query): Search the web

Use this format:
Thought: [your reasoning]
Action: tool_name(arguments)

When done:
Thought: [your reasoning]
Answer: [final response]
"""

    def _call_llm(self):
        messages = [{"role": "system", "content": self.system}]
        messages.extend(self.messages)
        completion = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0
        )
        return completion.choices[0].message.content`,
    explanation: `<strong>System Prompt:</strong> Defines the agent's behavior and output format. The ReAct format (Thought/Action/Answer) gives us a predictable structure to parse. Tool documentation tells the model what's available.`
  },
  3: {
    code: `from openai import OpenAI

class Agent:
    def __init__(self):
        self.client = OpenAI()
        self.messages = []
        self.system = """..."""  # System prompt from step 2

        # Tool registry - maps names to functions
        self.tools = {
            "calculate": self._calculate,
            "weather": self._weather,
            "search": self._search,
        }

    def _calculate(self, expression):
        """Evaluate a math expression."""
        try:
            return str(eval(expression))
        except Exception as e:
            return f"Error: {e}"

    def _weather(self, city):
        """Get weather for a city."""
        # In production, call a real API
        weathers = {"Paris": "72°F, sunny", "Tokyo": "65°F, cloudy"}
        return weathers.get(city, f"Weather data not available for {city}")

    def _search(self, query):
        """Search the web."""
        return f"Top result for '{query}': Wikipedia article about {query}"

    def _call_llm(self):
        # ... same as before`,
    explanation: `<strong>Tool Registry:</strong> A simple dictionary mapping tool names to functions. Each tool is just a Python function that takes arguments and returns a string. The agent will call these based on LLM decisions.`
  },
  4: {
    code: `import re
from openai import OpenAI

class Agent:
    # ... init and tools from previous steps

    def _parse_action(self, response):
        """Extract action and arguments from LLM response."""
        match = re.search(r"Action:\\s*(\\w+)\\((.+?)\\)", response, re.IGNORECASE)
        if match:
            action = match.group(1).lower()
            args = match.group(2).strip()
            return (action, args)
        return None

    def _parse_answer(self, response):
        """Extract final answer from LLM response."""
        match = re.search(r"Answer:\\s*(.+)", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def _execute_tool(self, action, args):
        """Execute a tool and return result."""
        if action not in self.tools:
            return f"Error: Unknown tool '{action}'. Available: {list(self.tools.keys())}"
        try:
            return self.tools[action](args)
        except Exception as e:
            return f"Error: {e}"`,
    explanation: `<strong>Response Parser:</strong> Uses regex to extract actions from the LLM output. We look for patterns like <code>Action: calculate(2+2)</code> and <code>Answer: The result is 4</code>. The execute method looks up tools in our registry.`
  },
  5: {
    code: `import re
from openai import OpenAI

class Agent:
    # ... all previous code

    def run(self, user_message, max_turns=10):
        """Run the agent until it produces an answer."""
        self.messages.append({"role": "user", "content": user_message})

        for turn in range(max_turns):
            # OBSERVE + THINK: Call the LLM
            response = self._call_llm()
            self.messages.append({"role": "assistant", "content": response})

            # Check for final answer
            answer = self._parse_answer(response)
            if answer:
                return answer

            # ACT: Parse and execute action
            action_result = self._parse_action(response)
            if action_result:
                action, args = action_result
                observation = self._execute_tool(action, args)

                # REFLECT: Add observation to history
                self.messages.append({
                    "role": "user",
                    "content": f"Observation: {observation}"
                })
            else:
                # No action or answer - prompt for format
                self.messages.append({
                    "role": "user",
                    "content": "Please use Action: or Answer: format."
                })

        return "Max turns reached without answer."

# Usage
agent = Agent()
result = agent.run("What is 25 * 17?")
print(result)  # "The answer is 425"`,
    explanation: `<strong>Execution Loop:</strong> The heart of the agent. We loop through: call LLM, check for answer, parse action, execute tool, add observation. This implements OBSERVE-THINK-ACT-REFLECT. Stop when we get an answer or hit max turns.`
  }
};

let currentBuildStep = 1;

function initBuildPanel() {
  const steps = document.querySelectorAll('.build-step');
  const resetBtn = document.getElementById('reset-build-btn');

  steps.forEach(step => {
    step.addEventListener('click', () => {
      const stepNum = parseInt(step.dataset.step);
      activateBuildStep(stepNum);
    });
  });

  resetBtn.addEventListener('click', resetBuild);
}

function activateBuildStep(stepNum) {
  const steps = document.querySelectorAll('.build-step');
  const codeEl = document.getElementById('build-code');
  const explanationEl = document.getElementById('build-explanation');
  const statusEl = document.getElementById('build-status');

  // Update step states
  steps.forEach(step => {
    const num = parseInt(step.dataset.step);
    step.classList.remove('active');

    if (num < stepNum) {
      step.classList.add('completed');
    } else if (num === stepNum) {
      step.classList.add('active');
      step.classList.add('completed');
    } else {
      step.classList.remove('completed');
    }
  });

  currentBuildStep = stepNum;

  // Update code display
  const stepData = buildSteps[stepNum];
  codeEl.textContent = stepData.code;
  explanationEl.innerHTML = stepData.explanation;
  statusEl.textContent = `Step ${stepNum} of 5`;
}

function resetBuild() {
  const steps = document.querySelectorAll('.build-step');
  steps.forEach((step, i) => {
    step.classList.remove('completed', 'active');
    if (i === 0) {
      step.classList.add('completed');
    }
  });

  currentBuildStep = 1;
  const stepData = buildSteps[1];
  document.getElementById('build-code').textContent = stepData.code;
  document.getElementById('build-explanation').innerHTML = stepData.explanation;
  document.getElementById('build-status').textContent = 'Step 1 of 5';
}

// =============================================================================
// PANEL 2: RUN AGENT
// =============================================================================

const queryExamples = {
  math: "What is 25 * 17?",
  weather: "What's the weather like in Paris?",
  multi: "What is 25 * 17, and what's the weather in Tokyo?"
};

// Simulated agent execution traces
const executionTraces = {
  math: [
    { phase: 'observe', content: 'User query: "What is 25 * 17?"' },
    { phase: 'think', content: 'Thought: I need to calculate 25 * 17. I\'ll use the calculate tool.' },
    { phase: 'act', content: 'Action: <code>calculate(25 * 17)</code>' },
    { phase: 'reflect', content: 'Observation: 425' },
    { phase: 'think', content: 'Thought: I have the answer now.' },
    { phase: 'answer', content: 'Answer: 25 * 17 = 425' }
  ],
  weather: [
    { phase: 'observe', content: 'User query: "What\'s the weather like in Paris?"' },
    { phase: 'think', content: 'Thought: I need to check the weather in Paris. I\'ll use the weather tool.' },
    { phase: 'act', content: 'Action: <code>weather(Paris)</code>' },
    { phase: 'reflect', content: 'Observation: 72°F, sunny' },
    { phase: 'think', content: 'Thought: I have the weather information.' },
    { phase: 'answer', content: 'Answer: The weather in Paris is currently 72°F and sunny.' }
  ],
  multi: [
    { phase: 'observe', content: 'User query: "What is 25 * 17, and what\'s the weather in Tokyo?"' },
    { phase: 'think', content: 'Thought: I have two tasks. First, let me calculate 25 * 17.' },
    { phase: 'act', content: 'Action: <code>calculate(25 * 17)</code>' },
    { phase: 'reflect', content: 'Observation: 425' },
    { phase: 'think', content: 'Thought: Got the math result. Now I need to check Tokyo\'s weather.' },
    { phase: 'act', content: 'Action: <code>weather(Tokyo)</code>' },
    { phase: 'reflect', content: 'Observation: 65°F, cloudy' },
    { phase: 'think', content: 'Thought: I have both pieces of information.' },
    { phase: 'answer', content: 'Answer: 25 * 17 = 425, and the weather in Tokyo is 65°F and cloudy.' }
  ]
};

let currentQuery = 'multi';
let agentState = {
  messages: [],
  toolCalls: 0,
  turns: 0,
  status: 'Idle',
  lastAction: null,
  lastObservation: null
};

function initRunPanel() {
  const runBtn = document.getElementById('run-agent-btn');
  const queryBtns = document.querySelectorAll('.query-btn');
  const maxTurnsInput = document.getElementById('max-turns');

  runBtn.addEventListener('click', runAgent);

  queryBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const query = btn.dataset.query;
      loadQuery(query);
    });
  });

  maxTurnsInput.addEventListener('change', () => {
    document.getElementById('max-turn-display').textContent = maxTurnsInput.value;
  });
}

function loadQuery(queryType) {
  currentQuery = queryType;
  document.getElementById('agent-query').value = queryExamples[queryType];
}

async function runAgent() {
  const logEl = document.getElementById('execution-log');
  const finalResultEl = document.getElementById('final-result');
  const slowMode = document.getElementById('slow-mode').checked;
  const delay = slowMode ? 800 : 200;

  // Reset UI
  logEl.innerHTML = '';
  finalResultEl.classList.remove('visible');
  finalResultEl.innerHTML = '';
  resetPhases();
  resetAgentState();

  const trace = executionTraces[currentQuery] || executionTraces.math;
  let turnCount = 0;

  for (const step of trace) {
    // Update phase highlighting
    activatePhase(step.phase);

    // Update turn count for act phases
    if (step.phase === 'act') {
      turnCount++;
      document.getElementById('turn-count').textContent = turnCount;
      agentState.toolCalls++;
    }

    // Add log entry
    addLogEntry(step.phase, step.content);

    // Update agent state
    updateAgentState(step);

    await sleep(delay);
  }

  // Show final result
  const answerStep = trace.find(s => s.phase === 'answer');
  if (answerStep) {
    finalResultEl.innerHTML = `
      <div class="final-result-header">Agent Complete</div>
      <div class="final-result-content">${answerStep.content.replace('Answer: ', '')}</div>
    `;
    finalResultEl.classList.add('visible');
  }

  agentState.status = 'Complete';
  updateInspectPanel();
}

function activatePhase(phase) {
  const phases = document.querySelectorAll('.loop-phase');
  phases.forEach(p => p.classList.remove('active'));

  // Map phase to element
  const phaseMap = {
    'observe': 'phase-observe',
    'think': 'phase-think',
    'act': 'phase-act',
    'reflect': 'phase-reflect',
    'answer': 'phase-think'
  };

  const phaseId = phaseMap[phase];
  if (phaseId) {
    document.getElementById(phaseId).classList.add('active');
  }
}

function resetPhases() {
  const phases = document.querySelectorAll('.loop-phase');
  phases.forEach(p => p.classList.remove('active'));
  document.getElementById('turn-count').textContent = '0';
}

function addLogEntry(phase, content) {
  const logEl = document.getElementById('execution-log');
  const entry = document.createElement('div');
  entry.className = `log-entry ${phase}`;
  entry.innerHTML = `
    <div class="log-phase">${phase}</div>
    <div class="log-content">${content}</div>
  `;
  logEl.appendChild(entry);
  logEl.scrollTop = logEl.scrollHeight;
}

function updateAgentState(step) {
  agentState.status = 'Running';

  if (step.phase === 'observe') {
    agentState.messages.push({ role: 'user', content: step.content });
  } else if (step.phase === 'think' || step.phase === 'answer') {
    agentState.messages.push({ role: 'assistant', content: step.content });
  } else if (step.phase === 'act') {
    agentState.lastAction = step.content;
    agentState.turns++;
  } else if (step.phase === 'reflect') {
    agentState.lastObservation = step.content;
    agentState.messages.push({ role: 'observation', content: step.content });
  }

  updateInspectPanel();
}

function resetAgentState() {
  agentState = {
    messages: [],
    toolCalls: 0,
    turns: 0,
    status: 'Running',
    lastAction: null,
    lastObservation: null
  };
  updateInspectPanel();
}

// =============================================================================
// PANEL 3: INSPECT INTERNALS
// =============================================================================

let messageFilter = 'all';

function initInspectPanel() {
  const filterBtns = document.querySelectorAll('.filter-btn');
  const clearBtn = document.getElementById('clear-state-btn');

  filterBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      filterBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      messageFilter = btn.dataset.filter;
      renderMessageHistory();
    });
  });

  clearBtn.addEventListener('click', () => {
    resetAgentState();
    document.getElementById('execution-log').innerHTML = `
      <div class="log-placeholder">Click "Run Agent" to start execution...</div>
    `;
    document.getElementById('final-result').classList.remove('visible');
    resetPhases();
  });
}

function updateInspectPanel() {
  // Update state values
  document.getElementById('state-messages').textContent = agentState.messages.length;
  document.getElementById('state-tool-calls').textContent = agentState.toolCalls;
  document.getElementById('state-turns').textContent = agentState.turns;
  document.getElementById('state-status').textContent = agentState.status;

  // Update last action
  const lastActionEl = document.getElementById('last-action');
  if (agentState.lastAction) {
    lastActionEl.innerHTML = agentState.lastAction;
  } else {
    lastActionEl.innerHTML = '<div class="no-action">No action yet</div>';
  }

  // Update last observation
  const lastObsEl = document.getElementById('last-observation');
  if (agentState.lastObservation) {
    lastObsEl.innerHTML = agentState.lastObservation;
  } else {
    lastObsEl.innerHTML = '<div class="no-observation">No observation yet</div>';
  }

  // Render message history
  renderMessageHistory();
}

function renderMessageHistory() {
  const listEl = document.getElementById('message-list');

  if (agentState.messages.length === 0) {
    listEl.innerHTML = `
      <div class="empty-history">
        Run an agent query to see message history here.
      </div>
    `;
    return;
  }

  let filtered = agentState.messages;
  if (messageFilter !== 'all') {
    filtered = agentState.messages.filter(m => {
      if (messageFilter === 'user') return m.role === 'user';
      if (messageFilter === 'assistant') return m.role === 'assistant' || m.role === 'observation';
      return true;
    });
  }

  listEl.innerHTML = filtered.map(m => {
    const roleClass = m.role === 'observation' ? 'observation' : m.role;
    return `
      <div class="message-item ${roleClass}">
        <div class="message-role">${m.role}</div>
        <div class="message-content">${escapeHtml(m.content)}</div>
      </div>
    `;
  }).join('');
}

// =============================================================================
// UTILITIES
// =============================================================================

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}
