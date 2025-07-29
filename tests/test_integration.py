"""
Integration tests for the AI Help Desk System.
"""
import pytest
import tempfile
import os
import json
from datetime import datetime
from pathlib import Path

# Import system components
from app import HelpDeskSystem
from src.models.schemas import HelpDeskRequest, RequestCategory
from config.settings import settings


class TestHelpDeskIntegration:
    """Integration tests for the complete help desk system."""
    
    @pytest.fixture
    def temp_knowledge_base(self):
        """Create a temporary knowledge base for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test documents
            kb_path = Path(temp_dir) / "knowledge_base"
            kb_path.mkdir()
            
            # Create FAQ document
            faq_content = """# Password Reset FAQ

## How to reset your password

1. Go to the company password reset portal
2. Enter your email address
3. Check your email for the reset link
4. Follow the instructions to create a new password

## Common Issues

- Account lockout: Contact IT support
- Email not received: Check spam folder
- Portal not working: Call the help desk
"""
            
            with open(kb_path / "password_faq.md", "w") as f:
                f.write(faq_content)
            
            # Create procedure document
            procedure_content = """# Software Installation Procedure

## Before Installing Software

1. Check if you have administrative privileges
2. Verify system requirements
3. Ensure the software is approved

## Installation Steps

1. Download from approved sources only
2. Run the installer as administrator
3. Follow the installation wizard
4. Restart if required

## Troubleshooting

- Permission denied: Contact IT for admin rights
- Installation fails: Check system compatibility
"""
            
            with open(kb_path / "software_installation.md", "w") as f:
                f.write(procedure_content)
            
            # Create JSON policy document
            policy_data = {
                "documents": [
                    {
                        "id": "security_policy_001",
                        "title": "Email Security Policy",
                        "content": "Never click on suspicious links in emails. Report phishing attempts to the security team immediately. Use strong passwords and enable two-factor authentication.",
                        "category": "security_incident",
                        "tags": ["security", "email", "phishing"]
                    }
                ]
            }
            
            with open(kb_path / "security_policies.json", "w") as f:
                json.dump(policy_data, f)
            
            # Update settings to use temp directory
            original_path = settings.system_config.knowledge_base_path
            settings.system_config.knowledge_base_path = str(kb_path)
            
            yield kb_path
            
            # Restore original path
            settings.system_config.knowledge_base_path = original_path
    
    @pytest.fixture
    def helpdesk_system(self, temp_knowledge_base):
        """Initialize help desk system with test data."""
        system = HelpDeskSystem()
        success = system.initialize_sync()
        assert success, "Failed to initialize help desk system"
        return system
    
    def test_password_reset_request(self, helpdesk_system):
        """Test processing a password reset request."""
        request_text = "I forgot my password and can't log into my computer. How do I reset it?"
        
        response = helpdesk_system.process_request_sync(request_text)
        
        # Assertions
        assert response is not None
        assert response.category == RequestCategory.PASSWORD_RESET
        assert response.confidence > 0.5
        assert "password" in response.response.lower()
        assert "reset" in response.response.lower()
        assert not response.escalation.should_escalate  # Should be self-service
        
    def test_software_installation_request(self, helpdesk_system):
        """Test processing a software installation request."""
        request_text = "I need to install Microsoft Office on my new laptop. Can you help?"
        
        response = helpdesk_system.process_request_sync(request_text)
        
        # Assertions
        assert response is not None
        assert response.category == RequestCategory.SOFTWARE_INSTALLATION
        assert response.confidence > 0.5
        assert "install" in response.response.lower()
        assert len(response.knowledge_sources) > 0  # Should find relevant documents
        
    def test_security_incident_escalation(self, helpdesk_system):
        """Test that security incidents are properly escalated."""
        request_text = "I received a suspicious email asking for my login credentials. It looks like phishing."
        
        response = helpdesk_system.process_request_sync(request_text)
        
        # Assertions
        assert response is not None
        assert response.category == RequestCategory.SECURITY_INCIDENT
        assert response.escalation.should_escalate  # Security should always escalate
        assert response.escalation.priority.value in ["high", "critical"]
        assert "security" in response.response.lower()
        
    def test_unknown_request_handling(self, helpdesk_system):
        """Test handling of unclear or unknown requests."""
        request_text = "Something is broken and I don't know what to do."
        
        response = helpdesk_system.process_request_sync(request_text)
        
        # Assertions
        assert response is not None
        assert response.confidence < 0.7  # Should have low confidence
        assert response.escalation.should_escalate  # Unknown issues should escalate
        assert "escalat" in response.response.lower() or "support" in response.response.lower()
        
    def test_network_connectivity_request(self, helpdesk_system):
        """Test processing a network connectivity request."""
        request_text = "My internet is not working and I can't access any websites."
        
        response = helpdesk_system.process_request_sync(request_text)
        
        # Assertions
        assert response is not None
        assert response.category == RequestCategory.NETWORK_CONNECTIVITY
        assert response.confidence > 0.5
        assert any(word in response.response.lower() for word in ["network", "internet", "connection"])
        
    def test_hardware_failure_request(self, helpdesk_system):
        """Test processing a hardware failure request."""
        request_text = "My printer is not working. It shows an error message and won't print anything."
        
        response = helpdesk_system.process_request_sync(request_text)
        
        # Assertions
        assert response is not None
        assert response.category == RequestCategory.HARDWARE_FAILURE
        assert response.confidence > 0.5
        assert "printer" in response.response.lower() or "hardware" in response.response.lower()
        
    def test_batch_processing(self, helpdesk_system):
        """Test batch processing of multiple requests."""
        requests = [
            {"request": "I forgot my password", "user_id": "user1"},
            {"request": "Need to install software", "user_id": "user2"},
            {"request": "Suspicious email received", "user_id": "user3"}
        ]
        
        responses = helpdesk_system.batch_process_requests(requests)
        
        # Assertions
        assert len(responses) == 3
        assert all(response is not None for response in responses)
        assert responses[0].category == RequestCategory.PASSWORD_RESET
        assert responses[1].category == RequestCategory.SOFTWARE_INSTALLATION
        assert responses[2].category == RequestCategory.SECURITY_INCIDENT
        
    def test_knowledge_base_retrieval(self, helpdesk_system):
        """Test that relevant knowledge base documents are retrieved."""
        request_text = "How do I reset my password?"
        
        response = helpdesk_system.process_request_sync(request_text)
        
        # Assertions
        assert response is not None
        assert len(response.knowledge_sources) > 0
        assert any("password" in source.lower() for source in response.knowledge_sources)
        
    def test_escalation_logic(self, helpdesk_system):
        """Test escalation logic for different types of requests."""
        # High confidence, good knowledge retrieval - should not escalate
        simple_request = "I forgot my password"
        simple_response = helpdesk_system.process_request_sync(simple_request)
        
        # Complex request - should escalate
        complex_request = "My server crashed and I lost all my data. The backup system also failed."
        complex_response = helpdesk_system.process_request_sync(complex_request)
        
        # Assertions
        assert not simple_response.escalation.should_escalate
        assert complex_response.escalation.should_escalate
        
    def test_response_quality(self, helpdesk_system):
        """Test that generated responses meet quality standards."""
        request_text = "How do I install Microsoft Office?"
        
        response = helpdesk_system.process_request_sync(request_text)
        
        # Assertions
        assert response is not None
        assert len(response.response) > 50  # Substantial response
        assert len(response.response) < 2000  # Not too long
        assert "install" in response.response.lower()
        assert response.response.endswith(('.', '!', '?'))  # Proper sentence ending
        
    def test_system_status(self, helpdesk_system):
        """Test system status reporting."""
        status = helpdesk_system.get_system_status()
        
        # Assertions
        assert status is not None
        assert status["initialized"] is True
        assert "knowledge_base" in status
        assert "workflow_metrics" in status
        assert "configuration" in status
        assert status["knowledge_base"]["document_count"] > 0
        
    def test_knowledge_base_refresh(self, helpdesk_system, temp_knowledge_base):
        """Test knowledge base refresh functionality."""
        # Add a new document
        new_doc_content = """# Email Configuration Guide

## Setting up Outlook

1. Open Outlook
2. Go to File > Add Account
3. Enter your email address
4. Follow the setup wizard

## Troubleshooting

- Can't connect: Check server settings
- Authentication failed: Verify password
"""
        
        with open(temp_knowledge_base / "email_config.md", "w") as f:
            f.write(new_doc_content)
        
        # Refresh knowledge base
        success = helpdesk_system.refresh_knowledge_base()
        assert success
        
        # Test that new document is available
        request_text = "How do I set up Outlook email?"
        response = helpdesk_system.process_request_sync(request_text)
        
        assert response.category == RequestCategory.EMAIL_CONFIGURATION
        assert "outlook" in response.response.lower()
        
    def test_error_handling(self, helpdesk_system):
        """Test system behavior with various error conditions."""
        # Empty request
        response = helpdesk_system.process_request_sync("")
        assert response is not None
        assert response.escalation.should_escalate
        
        # Very long request
        long_request = "help me " * 1000
        response = helpdesk_system.process_request_sync(long_request)
        assert response is not None
        
        # Special characters
        special_request = "My computer has ñ, é, ü characters and symbols: @#$%^&*()"
        response = helpdesk_system.process_request_sync(special_request)
        assert response is not None
        
    def test_classification_confidence_thresholds(self, helpdesk_system):
        """Test that classification confidence affects escalation decisions."""
        # Clear request - should have high confidence
        clear_request = "I forgot my password and need to reset it"
        clear_response = helpdesk_system.process_request_sync(clear_request)
        
        # Ambiguous request - should have lower confidence
        ambiguous_request = "Something is wrong with my thing"
        ambiguous_response = helpdesk_system.process_request_sync(ambiguous_request)
        
        # Assertions
        assert clear_response.confidence > ambiguous_response.confidence
        assert not clear_response.escalation.should_escalate
        assert ambiguous_response.escalation.should_escalate
        
    def test_category_specific_responses(self, helpdesk_system):
        """Test that responses are appropriate for each category."""
        test_cases = [
            {
                "request": "I forgot my password",
                "expected_category": RequestCategory.PASSWORD_RESET,
                "expected_keywords": ["password", "reset", "portal"]
            },
            {
                "request": "Need to install software",
                "expected_category": RequestCategory.SOFTWARE_INSTALLATION,
                "expected_keywords": ["install", "software", "admin"]
            },
            {
                "request": "Suspicious email with phishing",
                "expected_category": RequestCategory.SECURITY_INCIDENT,
                "expected_keywords": ["security", "phishing", "report"]
            }
        ]
        
        for test_case in test_cases:
            response = helpdesk_system.process_request_sync(test_case["request"])
            
            assert response.category == test_case["expected_category"]
            response_lower = response.response.lower()
            assert any(keyword in response_lower for keyword in test_case["expected_keywords"])
        
    def test_processing_time_performance(self, helpdesk_system):
        """Test that requests are processed within reasonable time limits."""
        request_text = "I need help with my computer"
        
        response = helpdesk_system.process_request_sync(request_text)
        
        # Assertions
        assert response.processing_time > 0
        assert response.processing_time < 30  # Should complete within 30 seconds
        
    def test_concurrent_requests(self, helpdesk_system):
        """Test system behavior with multiple concurrent requests."""
        import threading
        import queue
        
        requests = [
            "I forgot my password",
            "Need to install Office",
            "Internet not working",
            "Printer broken",
            "Suspicious email"
        ]
        
        results = queue.Queue()
        
        def process_request(req_text):
            try:
                response = helpdesk_system.process_request_sync(req_text)
                results.put(("success", response))
            except Exception as e:
                results.put(("error", str(e)))
        
        # Start threads
        threads = []
        for req in requests:
            thread = threading.Thread(target=process_request, args=(req,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=60)  # 60 second timeout
        
        # Check results
        successful_responses = 0
        while not results.empty():
            status, result = results.get()
            if status == "success":
                successful_responses += 1
                assert result is not None
        
        assert successful_responses == len(requests)


class TestSystemComponents:
    """Test individual system components."""
    
    def test_classification_agent(self):
        """Test the classification agent independently."""
        from src.agents.classifier_agent import classifier_agent
        from src.models.schemas import HelpDeskRequest
        
        request = HelpDeskRequest(
            id="test_001",
            request="I forgot my password and can't log in"
        )
        
        result = classifier_agent.classify_request(request)
        
        assert result.success
        assert result.data.category == RequestCategory.PASSWORD_RESET
        assert result.data.confidence > 0.5
        assert len(result.data.keywords) > 0
        
    def test_knowledge_agent(self):
        """Test the knowledge agent independently."""
        from src.agents.knowledge_agent import knowledge_agent
        from src.agents.classifier_agent import classifier_agent
        from src.models.schemas import HelpDeskRequest
        
        request = HelpDeskRequest(
            id="test_002",
            request="How do I reset my password?"
        )
        
        classification_result = classifier_agent.classify_request(request)
        assert classification_result.success
        
        retrieval_result = knowledge_agent.retrieve_knowledge(
            request, classification_result.data
        )
        
        assert retrieval_result.success
        assert retrieval_result.data.total_results >= 0
        
    def test_response_agent(self):
        """Test the response agent independently."""
        from src.agents.response_agent import response_agent
        from src.agents.classifier_agent import classifier_agent
        from src.agents.knowledge_agent import knowledge_agent
        from src.models.schemas import HelpDeskRequest
        
        request = HelpDeskRequest(
            id="test_003",
            request="I need help with password reset"
        )
        
        classification_result = classifier_agent.classify_request(request)
        retrieval_result = knowledge_agent.retrieve_knowledge(
            request, classification_result.data
        )
        
        response_result = response_agent.generate_response(
            request, classification_result.data, retrieval_result.data
        )
        
        assert response_result.success
        assert isinstance(response_result.data, str)
        assert len(response_result.data) > 20
        
    def test_escalation_agent(self):
        """Test the escalation agent independently."""
        from src.agents.escalation_agent import escalation_agent
        from src.agents.classifier_agent import classifier_agent
        from src.agents.knowledge_agent import knowledge_agent
        from src.models.schemas import HelpDeskRequest
        
        # Test security incident (should escalate)
        request = HelpDeskRequest(
            id="test_004",
            request="I received a phishing email"
        )
        
        classification_result = classifier_agent.classify_request(request)
        retrieval_result = knowledge_agent.retrieve_knowledge(
            request, classification_result.data
        )
        
        escalation_result = escalation_agent.determine_escalation(
            request, classification_result.data, retrieval_result.data
        )
        
        assert escalation_result.success
        assert escalation_result.data.should_escalate  # Security should escalate


# Utility functions for testing
def create_test_knowledge_base():
    """Create a minimal knowledge base for testing."""
    test_docs = [
        {
            "filename": "password_reset.md",
            "content": "# Password Reset\n\nTo reset your password, visit the company portal."
        },
        {
            "filename": "software_install.md",
            "content": "# Software Installation\n\nContact IT for software installation help."
        }
    ]
    return test_docs


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])