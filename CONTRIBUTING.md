# Contributing to MikoEcho

Thank you for your interest in contributing to MikoEcho! We welcome contributions from the community.

## 🌟 Ways to Contribute

- **Bug Reports**: Report bugs via GitHub Issues
- **Feature Requests**: Suggest new features or improvements
- **Code Contributions**: Submit pull requests for bug fixes or features
- **Documentation**: Improve docs, tutorials, or examples
- **Testing**: Help test new features and report issues

## 🚀 Getting Started

### 1. Fork the Repository

Click the "Fork" button on GitHub to create your own copy.

### 2. Clone Your Fork

```bash
git clone https://github.com/YOUR_USERNAME/MikoEcho.git
cd MikoEcho
```

### 3. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"
```

### 4. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

## 📝 Development Guidelines

### Code Style

We use:
- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

Format your code before committing:

```bash
black mikoecho/
isort mikoecho/
flake8 mikoecho/
mypy mikoecho/
```

### Testing

Write tests for new features:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=mikoecho tests/
```

### Commit Messages

Follow conventional commits:

```
feat: Add new emotion interpolation feature
fix: Resolve audio clipping issue in vocoder
docs: Update installation instructions
test: Add tests for voice cloner
```

## 🔄 Pull Request Process

1. **Update Documentation**: Ensure README and docs reflect your changes
2. **Add Tests**: Include tests for new functionality
3. **Format Code**: Run Black, isort, and linting
4. **Update CHANGELOG**: Add entry describing your changes
5. **Submit PR**: Create pull request with clear description

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
Describe testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] CHANGELOG updated
```

## 🐛 Reporting Bugs

Use GitHub Issues with:

- **Clear title** describing the issue
- **Steps to reproduce** the bug
- **Expected vs actual behavior**
- **Environment details** (OS, Python version, GPU)
- **Error messages** or logs

## 💡 Feature Requests

Submit feature requests via GitHub Issues:

- Describe the feature and use case
- Explain why it would be valuable
- Provide examples if possible

## 📚 Documentation

Help improve documentation:

- Fix typos or unclear explanations
- Add examples and tutorials
- Improve API documentation
- Translate to other languages

## 🤝 Code of Conduct

### Our Standards

- Be respectful and inclusive
- Welcome newcomers
- Accept constructive criticism
- Focus on what's best for the community

### Unacceptable Behavior

- Harassment or discrimination
- Trolling or insulting comments
- Publishing others' private information
- Unethical use of voice cloning technology

## ⚖️ Ethical Guidelines

Contributors must:

1. **Respect Consent**: Never contribute features that facilitate non-consensual voice cloning
2. **Promote Safety**: Consider safety implications of new features
3. **Transparency**: Clearly document capabilities and limitations
4. **Privacy**: Protect user data and privacy

## 📞 Questions?

- **GitHub Discussions**: For general questions
- **GitHub Issues**: For bug reports and feature requests
- **Email**: contact@artisticimpression.org

## 🙏 Recognition

Contributors will be recognized in:
- README.md acknowledgments
- Release notes
- Project documentation

Thank you for contributing to MikoEcho! 🎙️
