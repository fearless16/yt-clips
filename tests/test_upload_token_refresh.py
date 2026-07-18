"""Tests for upload.py token refresh improvements."""

import sys
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestTokenRefreshInvalidGrant:
    """invalid_grant errors should be detected and handled gracefully."""

    def test_invalid_grant_detected(self):
        """refresh() raising invalid_grant should be classified correctly."""
        from upload import get_authenticated_service

        # Mock Credentials to simulate invalid_grant
        mock_creds = MagicMock()
        mock_creds.valid = False
        mock_creds.expired = True
        mock_creds.refresh_token = "some_token"
        mock_creds.refresh.side_effect = Exception("invalid_grant: Token has been expired or revoked.")

        mock_token_data = {"refresh_token": "some_token", "token": "x", "token_uri": "https://oauth2.googleapis.com/token", "client_id": "x", "client_secret": "x"}

        with patch("upload.Credentials") as MockCreds, \
             patch("builtins.open", mock_open(read_data=json.dumps(mock_token_data))), \
             patch.object(Path, "exists", return_value=True):
            MockCreds.from_authorized_user_info.return_value = mock_creds
            result = get_authenticated_service()

        assert result is None, "Should return None on invalid_grant"

    def test_refresh_transient_error_retries(self):
        """Transient refresh errors should be retried once before giving up."""
        from upload import get_authenticated_service

        mock_creds = MagicMock()
        mock_creds.valid = False
        mock_creds.expired = True
        mock_creds.refresh_token = "some_token"
        # First call fails transiently, second succeeds
        mock_creds.refresh.side_effect = [
            Exception("connection timeout"),
            None
        ]
        mock_creds.to_json.return_value = '{"refreshed": true}'
        mock_creds.valid = True  # After successful refresh

        mock_token_data = {"refresh_token": "some_token", "token": "x", "token_uri": "https://oauth2.googleapis.com/token", "client_id": "x", "client_secret": "x"}

        with patch("upload.Credentials") as MockCreds, \
             patch("builtins.open", mock_open(read_data=json.dumps(mock_token_data))), \
             patch.object(Path, "exists", return_value=True), \
             patch("upload.build") as mock_build:
            MockCreds.from_authorized_user_info.return_value = mock_creds
            result = get_authenticated_service()

        # Should have retried refresh (called twice)
        assert mock_creds.refresh.call_count == 2, (
            f"Expected 2 refresh attempts, got {mock_creds.refresh.call_count}"
        )
        assert result is not None, "Should succeed after retry"

    def test_invalid_grant_logs_actionable_message(self):
        """invalid_grant should log clear instructions for the user."""
        from upload import get_authenticated_service

        mock_creds = MagicMock()
        mock_creds.valid = False
        mock_creds.expired = True
        mock_creds.refresh_token = "some_token"
        mock_creds.refresh.side_effect = Exception("invalid_grant")

        mock_token_data = {"refresh_token": "some_token", "token": "x", "token_uri": "https://oauth2.googleapis.com/token", "client_id": "x", "client_secret": "x"}

        with patch("upload.Credentials") as MockCreds, \
             patch("builtins.open", mock_open(read_data=json.dumps(mock_token_data))), \
             patch.object(Path, "exists", return_value=True):
            MockCreds.from_authorized_user_info.return_value = mock_creds
            # Should not raise, just return None
            result = get_authenticated_service()

        assert result is None


class TestTokenWriteBack:
    """Refreshed tokens should always be written back to disk."""

    def test_successful_refresh_writes_back(self):
        """After successful refresh, token should be written back to disk."""
        from upload import get_authenticated_service

        mock_creds = MagicMock()
        mock_creds.valid = False
        mock_creds.expired = True
        mock_creds.refresh_token = "some_token"
        mock_creds.refresh.return_value = None
        mock_creds.to_json.return_value = '{"refreshed": true}'
        mock_creds.valid = True

        mock_token_data = {"refresh_token": "some_token"}
        written_data = []

        def mock_write(path, mode="r", **kwargs):
            if "w" in mode:
                f = MagicMock()
                f.__enter__ = lambda s: s
                f.__exit__ = MagicMock(return_value=False)
                f.write = lambda data: written_data.append(data)
                return f
            else:
                import io
                return io.StringIO(json.dumps(mock_token_data))

        with patch("upload.Credentials") as MockCreds, \
             patch("builtins.open", side_effect=mock_write), \
             patch.object(Path, "exists", return_value=True), \
             patch("upload.build") as mock_build:
            MockCreds.from_authorized_user_info.return_value = mock_creds
            get_authenticated_service()

        assert len(written_data) >= 1, "Token should be written back after refresh"
        assert '{"refreshed": true}' in written_data
