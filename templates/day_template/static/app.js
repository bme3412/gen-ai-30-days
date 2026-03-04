/**
 * 30 Days of Agentic AI - Demo App Frontend
 * Handles vertical selection, API calls, and UI updates
 */

class AgentDemo {
    constructor() {
        // Get data from server
        this.day = window.APP_DATA.day;
        this.verticals = window.APP_DATA.verticals;
        this.selectedVertical = null;

        // DOM elements
        this.verticalCards = document.querySelectorAll('.vertical-card');
        this.demoInterface = document.getElementById('demo-interface');
        this.verticalSelector = document.querySelector('.vertical-selector');
        this.selectedEmoji = document.getElementById('selected-emoji');
        this.selectedName = document.getElementById('selected-name');
        this.changeVerticalBtn = document.getElementById('change-vertical');
        this.examplePrompts = document.getElementById('example-prompts');
        this.userInput = document.getElementById('user-input');
        this.sampleDataBtn = document.getElementById('toggle-sample-data');
        this.sampleDataPre = document.getElementById('sample-data');
        this.runAgentBtn = document.getElementById('run-agent');
        this.outputPlaceholder = document.getElementById('output-placeholder');
        this.outputContent = document.getElementById('output-content');
        this.outputLoading = document.getElementById('output-loading');
        this.outputSteps = document.getElementById('output-steps');
        this.outputResult = document.getElementById('output-result');
        this.copyOutputBtn = document.getElementById('copy-output');

        this.init();
    }

    init() {
        // Vertical card click handlers
        this.verticalCards.forEach(card => {
            card.addEventListener('click', () => {
                const verticalId = card.dataset.vertical;
                this.selectVertical(verticalId);
            });
        });

        // Change vertical button
        this.changeVerticalBtn.addEventListener('click', () => {
            this.resetVerticalSelection();
        });

        // Toggle sample data
        this.sampleDataBtn.addEventListener('click', () => {
            this.sampleDataPre.classList.toggle('hidden');
            const icon = this.sampleDataBtn.querySelector('i');
            if (this.sampleDataPre.classList.contains('hidden')) {
                icon.className = 'fas fa-database';
                this.sampleDataBtn.innerHTML = '<i class="fas fa-database"></i> View Sample Data';
            } else {
                icon.className = 'fas fa-eye-slash';
                this.sampleDataBtn.innerHTML = '<i class="fas fa-eye-slash"></i> Hide Sample Data';
            }
        });

        // Run agent button
        this.runAgentBtn.addEventListener('click', () => {
            this.runAgent();
        });

        // Copy output button
        this.copyOutputBtn.addEventListener('click', () => {
            this.copyOutput();
        });

        // Keyboard shortcut (Cmd/Ctrl + Enter to run)
        this.userInput.addEventListener('keydown', (e) => {
            if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
                this.runAgent();
            }
        });
    }

    selectVertical(verticalId) {
        // Find vertical data
        const vertical = this.verticals.find(v => v.id === verticalId);
        if (!vertical) return;

        this.selectedVertical = vertical;

        // Update UI
        this.verticalCards.forEach(card => {
            card.classList.remove('selected');
            if (card.dataset.vertical === verticalId) {
                card.classList.add('selected');
            }
        });

        // Show demo interface
        this.demoInterface.classList.remove('hidden');

        // Update header
        this.selectedEmoji.textContent = vertical.emoji;
        this.selectedName.textContent = vertical.name;

        // Populate example prompts
        this.examplePrompts.innerHTML = '';
        vertical.sample_prompts.forEach(prompt => {
            const chip = document.createElement('button');
            chip.className = 'prompt-chip';
            chip.textContent = prompt;
            chip.addEventListener('click', () => {
                this.userInput.value = prompt;
                this.userInput.focus();
            });
            this.examplePrompts.appendChild(chip);
        });

        // Update sample data
        this.sampleDataPre.textContent = JSON.stringify(vertical.sample_data, null, 2);

        // Reset output
        this.resetOutput();

        // Scroll to demo interface
        this.demoInterface.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    resetVerticalSelection() {
        this.selectedVertical = null;
        this.verticalCards.forEach(card => card.classList.remove('selected'));
        this.demoInterface.classList.add('hidden');
        this.verticalSelector.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    resetOutput() {
        this.outputPlaceholder.classList.remove('hidden');
        this.outputContent.classList.add('hidden');
        this.outputLoading.classList.add('hidden');
        this.outputSteps.innerHTML = '';
        this.outputResult.innerHTML = '';
    }

    async runAgent() {
        if (!this.selectedVertical) return;

        const input = this.userInput.value.trim();
        if (!input) {
            this.userInput.focus();
            return;
        }

        // Show loading state
        this.outputPlaceholder.classList.add('hidden');
        this.outputContent.classList.add('hidden');
        this.outputLoading.classList.remove('hidden');
        this.runAgentBtn.disabled = true;

        try {
            const response = await fetch('/api/run', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    vertical: this.selectedVertical.id,
                    input: input,
                    sample_data: this.selectedVertical.sample_data
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            this.displayOutput(data);

        } catch (error) {
            console.error('Error running agent:', error);
            this.displayError(error.message);
        } finally {
            this.runAgentBtn.disabled = false;
        }
    }

    displayOutput(data) {
        this.outputLoading.classList.add('hidden');
        this.outputContent.classList.remove('hidden');

        // Display steps (if any)
        this.outputSteps.innerHTML = '';
        if (data.steps && data.steps.length > 0) {
            data.steps.forEach((step, index) => {
                const stepEl = document.createElement('div');
                stepEl.className = 'output-step';
                stepEl.innerHTML = `
                    <div class="step-icon">${index + 1}</div>
                    <div class="step-content">
                        <div class="step-label">${step.type || 'Step'}</div>
                        <div class="step-text">${this.escapeHtml(step.content)}</div>
                    </div>
                `;
                this.outputSteps.appendChild(stepEl);
            });
        }

        // Display result
        this.outputResult.innerHTML = `
            <h4><i class="fas fa-check-circle"></i> Result</h4>
            <div class="output-result-content">${this.formatOutput(data.result)}</div>
        `;
    }

    displayError(message) {
        this.outputLoading.classList.add('hidden');
        this.outputContent.classList.remove('hidden');
        this.outputSteps.innerHTML = '';
        this.outputResult.innerHTML = `
            <h4 style="color: var(--error-color)"><i class="fas fa-exclamation-circle"></i> Error</h4>
            <div class="output-result-content">${this.escapeHtml(message)}</div>
        `;
        this.outputResult.style.borderLeftColor = 'var(--error-color)';
    }

    formatOutput(text) {
        if (!text) return '';
        // Convert markdown-style formatting to HTML
        let html = this.escapeHtml(text);
        // Convert **bold** to <strong>
        html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        // Convert newlines to <br>
        html = html.replace(/\n/g, '<br>');
        return html;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    copyOutput() {
        const resultContent = this.outputResult.querySelector('.output-result-content');
        if (resultContent) {
            const text = resultContent.innerText;
            navigator.clipboard.writeText(text).then(() => {
                // Visual feedback
                const icon = this.copyOutputBtn.querySelector('i');
                icon.className = 'fas fa-check';
                setTimeout(() => {
                    icon.className = 'fas fa-copy';
                }, 2000);
            });
        }
    }
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
    new AgentDemo();
});
