"""initial migration

Revision ID: initial_migration
Revises: 
Create Date: 2023-05-15 10:00:00.000000

"""
# pylint: disable=E1101
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'initial_migration'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Create chats table
    op.create_table('chats',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('title', sa.String(), nullable=True),
        sa.Column('created_at', sa.String(), nullable=True),
        sa.Column('focus_mode', sa.String(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_chats_id'), 'chats', ['id'], unique=False)
    op.create_index(op.f('ix_chats_title'), 'chats', ['title'], unique=False)

    # Create messages table
    op.create_table('messages',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('content', sa.Text(), nullable=True),
        sa.Column('chat_id', sa.String(), nullable=True),
        sa.Column('message_id', sa.String(), nullable=True),
        sa.Column('role', sa.String(), nullable=True),
        sa.Column('metadata', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['chat_id'], ['chats.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_messages_id'), 'messages', ['id'], unique=False)
    op.create_index(op.f('ix_messages_message_id'), 'messages', ['message_id'], unique=False)


def downgrade():
    op.drop_index(op.f('ix_messages_message_id'), table_name='messages')
    op.drop_index(op.f('ix_messages_id'), table_name='messages')
    op.drop_table('messages')
    op.drop_index(op.f('ix_chats_title'), table_name='chats')
    op.drop_index(op.f('ix_chats_id'), table_name='chats')
    op.drop_table('chats')
