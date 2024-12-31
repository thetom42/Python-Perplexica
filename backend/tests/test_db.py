import unittest
from datetime import datetime, timedelta
from sqlalchemy.exc import IntegrityError
from db.models import Chat, Message, RoleEnum
from db.crud import (
    create_chat,
    get_chats,
    get_chat,
    update_chat,
    delete_chat,
    create_message,
    get_chat_messages,
    bulk_create_messages,
    ChatNotFoundError,
    DatabaseError
)
from db.database import SessionLocal, Base, engine

class TestDB(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Create test database schema"""
        Base.metadata.create_all(engine)

    @classmethod
    def tearDownClass(cls):
        """Drop test database schema"""
        Base.metadata.drop_all(engine)

    def setUp(self):
        """Set up a new database session for each test"""
        self.connection = engine.connect()
        self.transaction = self.connection.begin()
        self.session = SessionLocal(bind=self.connection)

    def tearDown(self):
        """Rollback and close session after each test"""
        self.session.close()
        self.transaction.rollback()
        self.connection.close()

    def test_create_chat(self):
        """Test creating a new chat"""
        chat = create_chat(self.session, "Test Chat", False)
        self.assertIsNotNone(chat.id)
        self.assertEqual(chat.title, "Test Chat")
        self.assertFalse(chat.focus_mode)
        self.assertIsInstance(chat.created_at, datetime)

    def test_get_chats(self):
        """Test retrieving multiple chats"""
        # Create test data
        create_chat(self.session, "Chat 1", False)
        create_chat(self.session, "Chat 2", True)
        
        chats = get_chats(self.session)
        self.assertEqual(len(chats), 2)
        self.assertEqual(chats[0].title, "Chat 2")  # Should be ordered by created_at desc

    def test_get_chat(self):
        """Test retrieving a single chat"""
        chat = create_chat(self.session, "Test Chat", False)
        retrieved_chat = get_chat(self.session, chat.id)
        self.assertEqual(retrieved_chat.id, chat.id)

    def test_get_chat_not_found(self):
        """Test retrieving a non-existent chat"""
        with self.assertRaises(ChatNotFoundError):
            get_chat(self.session, "non-existent-id")

    def test_update_chat(self):
        """Test updating chat attributes"""
        chat = create_chat(self.session, "Test Chat", False)
        updated_chat = update_chat(self.session, chat.id, title="Updated Chat", focus_mode=True)
        self.assertEqual(updated_chat.title, "Updated Chat")
        self.assertTrue(updated_chat.focus_mode)

    def test_delete_chat(self):
        """Test deleting a chat"""
        chat = create_chat(self.session, "Test Chat", False)
        deleted_chat = delete_chat(self.session, chat.id)
        self.assertEqual(deleted_chat.id, chat.id)
        
        with self.assertRaises(ChatNotFoundError):
            get_chat(self.session, chat.id)

    def test_create_message(self):
        """Test creating a message in a chat"""
        chat = create_chat(self.session, "Test Chat", False)
        message = create_message(self.session, chat.id, "Test message", RoleEnum.USER)
        self.assertIsNotNone(message.id)
        self.assertEqual(message.content, "Test message")
        self.assertEqual(message.role, RoleEnum.USER)
        self.assertEqual(message.chat_id, chat.id)

    def test_get_chat_messages(self):
        """Test retrieving messages for a chat"""
        chat = create_chat(self.session, "Test Chat", False)
        create_message(self.session, chat.id, "Message 1", RoleEnum.USER)
        create_message(self.session, chat.id, "Message 2", RoleEnum.ASSISTANT)
        
        messages = get_chat_messages(self.session, chat.id)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0].content, "Message 2")  # Should be ordered by created_at desc

    def test_bulk_create_messages(self):
        """Test creating multiple messages in a single transaction"""
        chat = create_chat(self.session, "Test Chat", False)
        messages_data = [
            {"chat_id": chat.id, "content": "Message 1", "role": RoleEnum.USER},
            {"chat_id": chat.id, "content": "Message 2", "role": RoleEnum.ASSISTANT}
        ]
        
        messages = bulk_create_messages(self.session, messages_data)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0].content, "Message 1")

    def test_chat_model_validation(self):
        """Test Chat model validation"""
        with self.assertRaises(IntegrityError):
            chat = Chat(title=None)  # title is required
            self.session.add(chat)
            self.session.commit()

    def test_message_model_validation(self):
        """Test Message model validation"""
        chat = create_chat(self.session, "Test Chat", False)
        with self.assertRaises(IntegrityError):
            message = Message(chat_id=chat.id, content=None, role=RoleEnum.USER)  # content is required
            self.session.add(message)
            self.session.commit()

if __name__ == '__main__':
    unittest.main()