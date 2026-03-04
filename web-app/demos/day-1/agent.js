/**
 * Day 1 Demo: Research Agent with OBSERVE-THINK-ACT-REFLECT Loop
 *
 * This demonstrates the core agentic pattern using the OpenAI API.
 * The agent iteratively researches a topic by:
 * - OBSERVE: Analyzing the current state and what information is available
 * - THINK: Deciding what action to take next
 * - ACT: Executing the chosen action (generating content, searching, etc.)
 * - REFLECT: Evaluating if the task is complete or needs more work
 */

// ═══════════════════════════════════════════════════════════════
// AGENT STATE
// ═══════════════════════════════════════════════════════════════

let agentState = {
  isRunning: false,
  iteration: 0,
  maxIterations: 3,
  query: '',
  gatheredInfo: [],
  currentPhase: null,
  isComplete: false
};

// ═══════════════════════════════════════════════════════════════
// UI HELPERS
// ═══════════════════════════════════════════════════════════════

function setActivePhase(phase) {
  document.querySelectorAll('.phase-node').forEach(node => {
    node.classList.remove('active');
    if (node.dataset.phase === phase) {
      node.classList.add('active');
    }
  });
  agentState.currentPhase = phase;
}

function addStep(phase, content) {
  const log = document.getElementById('steps-log');
  if (log.querySelector('.empty-state')) {
    log.innerHTML = '';
  }

  const time = new Date().toLocaleTimeString();
  const step = document.createElement('div');
  step.className = 'step-entry';
  step.innerHTML = `
    <div class="step-header">
      <span class="step-phase ${phase}">${phase}</span>
      <span class="step-time">${time}</span>
    </div>
    <div class="step-content">${escapeHtml(content)}</div>
  `;
  log.appendChild(step);
  log.scrollTop = log.scrollHeight;
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function updateIterationBadge() {
  document.getElementById('iteration-badge').innerHTML =
    `Iteration <strong>${agentState.iteration}</strong> of ${agentState.maxIterations}`;
}

function showError(message) {
  addStep('error', message);
}

function showResult(content) {
  document.getElementById('result-content').textContent = content;
  document.getElementById('result-section').style.display = 'block';
}

function resetUI() {
  document.getElementById('steps-log').innerHTML = `
    <div class="empty-state">
      <div class="empty-state-icon">&#129302;</div>
      <p>Enter your API key and a research question to see the agent loop in action</p>
    </div>
  `;
  document.getElementById('result-section').style.display = 'none';
  document.getElementById('iteration-badge').textContent = 'Ready to start';
  document.querySelectorAll('.phase-node').forEach(n => {
    n.classList.remove('active', 'completed');
  });
}

// ═══════════════════════════════════════════════════════════════
// OPENAI API CALLS
// ═══════════════════════════════════════════════════════════════

async function callOpenAI(messages, jsonMode = false) {
  const apiKey = document.getElementById('api-key').value.trim();
  const model = document.getElementById('model').value;

  if (!apiKey) {
    throw new Error('Please enter your OpenAI API key');
  }

  const body = {
    model: model,
    messages: messages,
    temperature: 0.7,
    max_tokens: 1000
  };

  if (jsonMode) {
    body.response_format = { type: 'json_object' };
  }

  const response = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`
    },
    body: JSON.stringify(body)
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error?.message || 'API call failed');
  }

  const data = await response.json();
  return data.choices[0].message.content;
}

// ═══════════════════════════════════════════════════════════════
// AGENT LOOP PHASES
// ═══════════════════════════════════════════════════════════════

/**
 * OBSERVE: Analyze the current state
 * What do we know? What are we trying to accomplish?
 */
async function observe() {
  setActivePhase('observe');

  const systemPrompt = document.getElementById('agent-persona').value;

  const prompt = agentState.iteration === 1
    ? `You are observing the start of a research task.

USER QUERY: "${agentState.query}"

Analyze this query and identify:
1. The main topic/subject
2. Key aspects to research
3. What kind of information would be helpful

Be concise (2-3 sentences).`
    : `You are observing the current state of a research task.

USER QUERY: "${agentState.query}"

INFORMATION GATHERED SO FAR:
${agentState.gatheredInfo.map((info, i) => `${i + 1}. ${info}`).join('\n')}

Analyze what we have and identify:
1. What aspects have been covered
2. What gaps remain
3. What would strengthen the response

Be concise (2-3 sentences).`;

  const observation = await callOpenAI([
    { role: 'system', content: systemPrompt },
    { role: 'user', content: prompt }
  ]);

  addStep('observe', observation);
  return observation;
}

/**
 * THINK: Decide what action to take
 * Based on observation, what should we do next?
 */
async function think(observation) {
  setActivePhase('think');

  const prompt = `Based on this observation about our research task:

"${observation}"

QUERY: "${agentState.query}"
ITERATION: ${agentState.iteration} of ${agentState.maxIterations}

Decide what to do next. Choose ONE action:
- RESEARCH: Gather more specific information about an aspect
- SYNTHESIZE: Combine gathered information into insights
- ELABORATE: Expand on a particular point

Respond in JSON format:
{
  "action": "RESEARCH" | "SYNTHESIZE" | "ELABORATE",
  "focus": "what specific aspect to focus on",
  "reasoning": "brief explanation of why this action"
}`;

  const thinkResult = await callOpenAI([
    { role: 'user', content: prompt }
  ], true);

  const decision = JSON.parse(thinkResult);
  addStep('think', `Action: ${decision.action}\nFocus: ${decision.focus}\nReasoning: ${decision.reasoning}`);
  return decision;
}

/**
 * ACT: Execute the chosen action
 * Generate content based on the decision
 */
async function act(decision) {
  setActivePhase('act');

  const systemPrompt = document.getElementById('agent-persona').value;

  let prompt;
  switch (decision.action) {
    case 'RESEARCH':
      prompt = `Research and provide detailed information about: ${decision.focus}

Context: This is for answering the query "${agentState.query}"

Provide 2-3 key facts or insights. Be specific and informative.`;
      break;
    case 'SYNTHESIZE':
      prompt = `Synthesize the following information into a coherent insight:

${agentState.gatheredInfo.map((info, i) => `${i + 1}. ${info}`).join('\n')}

Focus on: ${decision.focus}
Create a unified understanding that addresses: "${agentState.query}"`;
      break;
    case 'ELABORATE':
      prompt = `Elaborate on this aspect: ${decision.focus}

Context: Answering "${agentState.query}"
Previous information: ${agentState.gatheredInfo.slice(-2).join('; ')}

Provide deeper insights or examples.`;
      break;
    default:
      prompt = `Provide helpful information about: ${decision.focus}`;
  }

  const result = await callOpenAI([
    { role: 'system', content: systemPrompt },
    { role: 'user', content: prompt }
  ]);

  agentState.gatheredInfo.push(result);
  addStep('act', `[${decision.action}]\n${result}`);
  return result;
}

/**
 * REFLECT: Evaluate progress and decide if complete
 * Should we continue or are we done?
 */
async function reflect() {
  setActivePhase('reflect');

  const prompt = `Evaluate the research progress for this task:

QUERY: "${agentState.query}"

GATHERED INFORMATION:
${agentState.gatheredInfo.map((info, i) => `${i + 1}. ${info}`).join('\n\n')}

ITERATION: ${agentState.iteration} of ${agentState.maxIterations}

Assess:
1. How well does the gathered information answer the query? (0-100%)
2. Is there enough information for a comprehensive response?
3. What's missing (if anything)?

Respond in JSON:
{
  "completeness": 0-100,
  "isComplete": true/false,
  "summary": "brief assessment",
  "missing": "what's still needed (or 'nothing' if complete)"
}`;

  const reflectResult = await callOpenAI([
    { role: 'user', content: prompt }
  ], true);

  const reflection = JSON.parse(reflectResult);
  addStep('reflect', `Completeness: ${reflection.completeness}%\n${reflection.summary}\n${reflection.isComplete ? '✓ Ready to finalize' : `Missing: ${reflection.missing}`}`);

  // Mark complete if reflection says so OR we've reached max iterations
  agentState.isComplete = reflection.isComplete || agentState.iteration >= agentState.maxIterations;

  return reflection;
}

/**
 * FINALIZE: Create the final response
 */
async function finalize() {
  const systemPrompt = document.getElementById('agent-persona').value;

  const prompt = `Create a final, well-structured response to this query:

QUERY: "${agentState.query}"

RESEARCHED INFORMATION:
${agentState.gatheredInfo.map((info, i) => `${i + 1}. ${info}`).join('\n\n')}

Synthesize this into a clear, comprehensive response. Use the information gathered but present it naturally, not as a list. Be helpful and informative.`;

  const finalResponse = await callOpenAI([
    { role: 'system', content: systemPrompt },
    { role: 'user', content: prompt }
  ]);

  return finalResponse;
}

// ═══════════════════════════════════════════════════════════════
// MAIN AGENT LOOP
// ═══════════════════════════════════════════════════════════════

async function runAgent() {
  if (agentState.isRunning) return;

  const query = document.getElementById('query').value.trim();
  const apiKey = document.getElementById('api-key').value.trim();

  if (!apiKey) {
    alert('Please enter your OpenAI API key');
    return;
  }

  if (!query) {
    alert('Please enter a research question');
    return;
  }

  // Initialize state
  agentState = {
    isRunning: true,
    iteration: 0,
    maxIterations: parseInt(document.getElementById('max-iterations').value),
    query: query,
    gatheredInfo: [],
    currentPhase: null,
    isComplete: false
  };

  // Update UI
  document.getElementById('run-btn').disabled = true;
  document.getElementById('run-btn').innerHTML = '<span class="loading"></span> Running...';
  document.getElementById('result-section').style.display = 'none';
  document.getElementById('steps-log').innerHTML = '';

  try {
    // Run the OTAR loop
    while (!agentState.isComplete && agentState.iteration < agentState.maxIterations) {
      agentState.iteration++;
      updateIterationBadge();

      // OBSERVE
      const observation = await observe();
      await sleep(500);

      // THINK
      const decision = await think(observation);
      await sleep(500);

      // ACT
      await act(decision);
      await sleep(500);

      // REFLECT
      await reflect();
      await sleep(500);
    }

    // Finalize
    document.getElementById('iteration-badge').innerHTML =
      `Completed in <strong>${agentState.iteration}</strong> iteration${agentState.iteration > 1 ? 's' : ''} - Generating final response...`;

    setActivePhase(null);
    document.querySelectorAll('.phase-node').forEach(n => n.classList.add('completed'));

    const finalResponse = await finalize();
    showResult(finalResponse);

    document.getElementById('iteration-badge').innerHTML =
      `Completed in <strong>${agentState.iteration}</strong> iteration${agentState.iteration > 1 ? 's' : ''}`;

  } catch (error) {
    showError(`Error: ${error.message}`);
    console.error(error);
  } finally {
    agentState.isRunning = false;
    document.getElementById('run-btn').disabled = false;
    document.getElementById('run-btn').innerHTML = '<span>&#9654;</span> Run Agent';
  }
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// ═══════════════════════════════════════════════════════════════
// EVENT LISTENERS
// ═══════════════════════════════════════════════════════════════

document.getElementById('run-btn').addEventListener('click', runAgent);
document.getElementById('reset-btn').addEventListener('click', () => {
  agentState = {
    isRunning: false,
    iteration: 0,
    maxIterations: 3,
    query: '',
    gatheredInfo: [],
    currentPhase: null,
    isComplete: false
  };
  resetUI();
});

// Allow Enter key to run
document.getElementById('query').addEventListener('keypress', (e) => {
  if (e.key === 'Enter' && !agentState.isRunning) {
    runAgent();
  }
});

// Save API key to localStorage
document.getElementById('api-key').addEventListener('change', (e) => {
  localStorage.setItem('openai_api_key', e.target.value);
});

// Load API key from localStorage on page load
window.addEventListener('DOMContentLoaded', () => {
  const savedKey = localStorage.getItem('openai_api_key');
  if (savedKey) {
    document.getElementById('api-key').value = savedKey;
  }
});
