# Astrabot Documentation

Welcome to the Astrabot documentation! Astrabot is a personal AI fine-tuning project that creates language models mimicking your communication style by analyzing Signal messenger conversation history.

## Documentation Structure

This documentation follows the [Diátaxis](https://diataxis.fr/) framework, organizing content into four distinct types:

### 🎓 Tutorials (Learning-Oriented)
Step-by-step guides to get you started with Astrabot.

- **[Getting Started](tutorials/getting-started.md)** - Complete setup and first model training
- **[Your First Training Run](tutorials/first-training-run.md)** - Detailed walkthrough of training process

### 🔧 How-To Guides (Task-Oriented)
Practical guides for specific tasks.

- **[Process Signal Backup](how-to/process-signal-backup.md)** - Extract data from Signal backups
- **[Process Conversations](how-to/process-conversations.md)** - Transform conversations into training data
- **[Deploy Your Model](how-to/deploy-model.md)** - Using your trained model in applications
- **[Advanced Training](how-to/advanced-training.md)** - Fine-tuning techniques and optimization

### 📖 Reference (Information-Oriented)
Technical reference for APIs, schemas, and utilities.

- **[API Reference](reference/api.md)** - Complete API documentation
- **[Signal Data Schema](reference/signal-data-schema.md)** - Database structure and relationships
- **[Utilities Reference](reference/utilities.md)** - Configuration and logging systems
- **[TrainingDataCreator API](reference/api/training-data-creator.md)** - Training data generation

### 💡 Explanation (Understanding-Oriented)
Conceptual guides explaining the why and how.

- **[Architecture Overview](explanation/architecture.md)** - System design and components
- **[Privacy Architecture](explanation/privacy-architecture.md)** - Comprehensive privacy and security guide
- **[Communication Styles](explanation/communication-styles.md)** - How Astrabot learns your style

## Quick Links

### For New Users
1. Start with the **[Getting Started Tutorial](tutorials/getting-started.md)**
2. Read about **[Privacy Considerations](explanation/privacy-considerations.md)**
3. Follow the **[First Training Run](tutorials/first-training-run.md)**

### For Developers
1. Review the **[Architecture](explanation/architecture.md)**
2. Check the **[API Reference](reference/api.md)**
3. Understand the **[Signal Data Schema](reference/signal-data-schema.md)**

### Common Tasks
- [Extract Signal backup data](how-to/process-signal-backup.md)
- [Configure environment variables](reference/environment-variables.md)
- [Set up development environment](CONTRIBUTING.md)
- [Run tests](../tests/README.md)

## Additional Resources

### Development
- **[Contributing Guide](../CONTRIBUTING.md)** - How to contribute to Astrabot
- **[Test-Driven Development](explanation/tdd-guide.md)** - TDD methodology and examples
- **[Environment Setup](how-to/pyenv-setup.md)** - Python environment configuration

### Project Information
- **[Changelog](../CHANGELOG.md)** - Version history and updates
- **[Project Structure](../PROJECT_STRUCTURE_ANALYSIS.md)** - Codebase organization
- **[GitHub Repository](https://github.com/yourusername/astrabot)** - Source code and issues

## Getting Help

### Support Channels
- **GitHub Issues** - Bug reports and feature requests
- **Discussions** - Community questions and answers
- **Documentation** - You are here! 

### Troubleshooting
- Check the troubleshooting section in **[Getting Started](tutorials/getting-started.md#troubleshooting)**
- Review logs in `data/logs/astrabot.log`
- Run diagnostics: `python scripts/diagnose.py`

### Security Notes
- Never share your Signal backup files
- Don't commit `.env` files or API keys
- Be cautious when sharing trained models (they may contain personal patterns)
- Review **[Privacy Considerations](explanation/privacy-considerations.md)** before deployment

## Documentation Improvements

Found an issue or want to improve the docs? 
- Open a PR with your changes
- File an issue describing what's unclear
- Suggest new guides or tutorials

---

*Last updated: {{ date }}*  
*Astrabot version: {{ version }}*