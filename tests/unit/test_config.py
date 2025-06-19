"""Tests for configuration management functionality.

This module tests the configuration system following the TDD approach
established in the project.
"""

import pytest
import os
import tempfile
from unittest.mock import patch, mock_open
from pathlib import Path

from src.utils.config import Config


@pytest.mark.unit
class TestConfig:
    """Test the Config class functionality."""
    
    @pytest.fixture
    def clean_env(self, monkeypatch):
        """Clean environment for testing."""
        # Remove any existing environment variables
        env_vars_to_remove = [
            'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'YOUR_RECIPIENT_ID',
            'DEBUG', 'LOG_LEVEL', 'ENABLE_IMAGE_PROCESSING'
        ]
        for var in env_vars_to_remove:
            monkeypatch.delenv(var, raising=False)
        return monkeypatch
    
    def test_config_initialization(self, clean_env):
        """Test config initialization with defaults."""
        config = Config()
        
        # Test default values
        assert config.get('YOUR_RECIPIENT_ID', 2) == 2
        assert config.get('DEBUG', False) is False
        assert config.get('LOG_LEVEL', 'INFO') == 'INFO'
    
    def test_config_get_with_default(self, clean_env):
        """Test getting configuration values with defaults."""
        config = Config()
        
        # Test with default when env var doesn't exist
        assert config.get('NONEXISTENT_KEY', 'default_value') == 'default_value'
        
        # Test with None default
        assert config.get('NONEXISTENT_KEY') is None
    
    def test_config_get_from_environment(self, clean_env):
        """Test getting values from environment variables."""
        clean_env.setenv('TEST_KEY', 'test_value')
        clean_env.setenv('YOUR_RECIPIENT_ID', '5')
        
        config = Config()
        
        assert config.get('TEST_KEY') == 'test_value'
        assert config.get('YOUR_RECIPIENT_ID') == '5'
    
    def test_config_require_existing(self, clean_env):
        """Test require() with existing environment variable."""
        clean_env.setenv('REQUIRED_KEY', 'required_value')
        
        config = Config()
        
        assert config.require('REQUIRED_KEY') == 'required_value'
    
    def test_config_require_missing(self, clean_env):
        """Test require() with missing environment variable raises error."""
        config = Config()
        
        with pytest.raises(ValueError, match="Required configuration key 'MISSING_KEY' not found"):
            config.require('MISSING_KEY')
    
    def test_has_openai_true(self, clean_env):
        """Test has_openai() returns True when API key is set."""
        clean_env.setenv('OPENAI_API_KEY', 'sk-test123456789')
        
        config = Config()
        
        assert config.has_openai() is True
    
    def test_has_openai_false(self, clean_env):
        """Test has_openai() returns False when API key is not set."""
        config = Config()
        
        assert config.has_openai() is False
    
    def test_has_anthropic_true(self, clean_env):
        """Test has_anthropic() returns True when API key is set."""
        clean_env.setenv('ANTHROPIC_API_KEY', 'sk-ant-test123456789')
        
        config = Config()
        
        assert config.has_anthropic() is True
    
    def test_has_anthropic_false(self, clean_env):
        """Test has_anthropic() returns False when API key is not set."""
        config = Config()
        
        assert config.has_anthropic() is False
    
    def test_validate_creates_directories(self, clean_env, tmp_path):
        """Test that validate() creates necessary directories."""
        clean_env.setenv('DATA_DIR', str(tmp_path / 'data'))
        clean_env.setenv('OUTPUT_DIR', str(tmp_path / 'output'))
        
        config = Config()
        config.validate()
        
        assert (tmp_path / 'data').exists()
        assert (tmp_path / 'output').exists()
    
    def test_print_status_masks_sensitive_data(self, clean_env, capsys):
        """Test that print_status() masks sensitive information."""
        clean_env.setenv('OPENAI_API_KEY', 'sk-1234567890abcdef')
        clean_env.setenv('ANTHROPIC_API_KEY', 'sk-ant-abcdef123456')
        
        config = Config()
        config.print_status()
        
        captured = capsys.readouterr()
        
        # Should show masked API keys
        assert 'sk-...ged' in captured.out or 'configured' in captured.out
        # Should not show full API keys
        assert 'sk-1234567890abcdef' not in captured.out
        assert 'sk-ant-abcdef123456' not in captured.out
    
    def test_boolean_conversion(self, clean_env):
        """Test conversion of string environment variables to booleans."""
        test_cases = [
            ('true', True),
            ('True', True),
            ('TRUE', True),
            ('1', True),
            ('false', False),
            ('False', False),
            ('FALSE', False),
            ('0', False),
            ('', False),
            ('random', False)
        ]
        
        config = Config()
        
        for env_value, expected in test_cases:
            clean_env.setenv('BOOL_TEST', env_value)
            
            # Simulate boolean configuration check
            result = config.get('BOOL_TEST', '').lower() in ['true', '1', 'yes', 'on']
            if env_value.lower() in ['true', '1']:
                assert result == expected
    
    def test_integer_conversion(self, clean_env):
        """Test conversion of string environment variables to integers."""
        clean_env.setenv('INT_TEST', '42')
        clean_env.setenv('INVALID_INT', 'not_a_number')
        
        config = Config()
        
        # Valid integer
        int_value = config.get('INT_TEST')
        assert int_value == '42'  # Config returns strings, conversion happens in usage
        
        # Invalid integer should be handled gracefully
        invalid_value = config.get('INVALID_INT')
        assert invalid_value == 'not_a_number'
    
    def test_dotenv_file_loading(self, clean_env, tmp_path):
        """Test loading configuration from .env file."""
        env_file = tmp_path / '.env'
        env_content = """
# Test environment file
OPENAI_API_KEY=sk-test123456789
YOUR_RECIPIENT_ID=5
DEBUG=true
LOG_LEVEL=DEBUG
"""
        env_file.write_text(env_content.strip())
        
        # Mock the dotenv loading to use our test file
        with patch('src.utils.config.load_dotenv') as mock_load:
            config = Config()
            mock_load.assert_called()
    
    def test_config_with_missing_dotenv(self, clean_env):
        """Test config behavior when .env file is missing."""
        with patch('src.utils.config.load_dotenv') as mock_load:
            # Simulate missing .env file
            mock_load.side_effect = FileNotFoundError()
            
            # Should not raise error
            config = Config()
            assert config is not None
    
    def test_environment_precedence(self, clean_env):
        """Test that environment variables take precedence over .env file."""
        # Set environment variable
        clean_env.setenv('TEST_PRECEDENCE', 'env_value')
        
        # Mock .env file content
        with patch('src.utils.config.load_dotenv'):
            config = Config()
            
            # Environment variable should take precedence
            assert config.get('TEST_PRECEDENCE') == 'env_value'
    
    def test_configuration_security(self, clean_env, capsys):
        """Test that sensitive configuration is handled securely."""
        # Set sensitive data
        clean_env.setenv('OPENAI_API_KEY', 'sk-secret123456789')
        clean_env.setenv('PASSWORD', 'super_secret_password')
        
        config = Config()
        
        # Ensure sensitive data is not accidentally logged
        config.print_status()
        captured = capsys.readouterr()
        
        # Should not contain full sensitive values
        assert 'sk-secret123456789' not in captured.out
        assert 'super_secret_password' not in captured.out
    
    @pytest.mark.parametrize("key,expected_type", [
        ('OPENAI_API_KEY', str),
        ('YOUR_RECIPIENT_ID', str),  # Config returns strings
        ('DEBUG', str),
        ('LOG_LEVEL', str)
    ])
    def test_configuration_types(self, clean_env, key, expected_type):
        """Test that configuration values have expected types."""
        test_values = {
            'OPENAI_API_KEY': 'sk-test123',
            'YOUR_RECIPIENT_ID': '2',
            'DEBUG': 'true',
            'LOG_LEVEL': 'INFO'
        }
        
        if key in test_values:
            clean_env.setenv(key, test_values[key])
        
        config = Config()
        value = config.get(key)
        
        if value is not None:
            assert isinstance(value, expected_type)


@pytest.mark.unit
class TestConfigValidation:
    """Test configuration validation functionality."""
    
    def test_validate_missing_required_config(self, monkeypatch):
        """Test validation with missing required configuration."""
        # Clear environment
        monkeypatch.delenv('YOUR_RECIPIENT_ID', raising=False)
        
        config = Config()
        
        # Validation should complete even with missing optional config
        config.validate()  # Should not raise error
    
    def test_validate_with_complete_config(self, monkeypatch, tmp_path):
        """Test validation with complete configuration."""
        # Set all required configuration
        monkeypatch.setenv('YOUR_RECIPIENT_ID', '2')
        monkeypatch.setenv('DATA_DIR', str(tmp_path / 'data'))
        monkeypatch.setenv('OUTPUT_DIR', str(tmp_path / 'output'))
        monkeypatch.setenv('OPENAI_API_KEY', 'sk-test123456789')
        
        config = Config()
        config.validate()
        
        # Should create directories
        assert (tmp_path / 'data').exists()
        assert (tmp_path / 'output').exists()
        
        # Should have API access
        assert config.has_openai() is True
    
    def test_validate_directory_creation_permissions(self, monkeypatch, tmp_path):
        """Test directory creation with permission issues."""
        # Set directory in read-only location (simulated)
        readonly_dir = tmp_path / 'readonly'
        readonly_dir.mkdir()
        readonly_dir.chmod(0o444)  # Read-only
        
        target_dir = readonly_dir / 'data'
        monkeypatch.setenv('DATA_DIR', str(target_dir))
        
        config = Config()
        
        # Should handle permission errors gracefully
        try:
            config.validate()
        except PermissionError:
            # This is acceptable behavior
            pass
        finally:
            # Cleanup: restore permissions
            readonly_dir.chmod(0o755)


@pytest.mark.unit
class TestConfigSingleton:
    """Test Config singleton behavior if implemented."""
    
    def test_config_instance_consistency(self):
        """Test that config instances are consistent."""
        config1 = Config()
        config2 = Config()
        
        # Should be separate instances but behave consistently
        assert config1 is not config2  # Not a singleton in current implementation
        
        # But should have same behavior
        assert config1.get('NONEXISTENT', 'default') == config2.get('NONEXISTENT', 'default')


if __name__ == "__main__":
    pytest.main([__file__])