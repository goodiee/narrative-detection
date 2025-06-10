import datetime
import pytz
from telethon import TelegramClient
import csv
import asyncio

API_ID = ''
API_HASH = ''

TELEGRAM_CHANNELS = [
    'rubaltic'
]

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

timezone = pytz.UTC

START_DATE = timezone.localize(datetime.datetime(2024, 1, 1))
END_DATE = timezone.localize(datetime.datetime(2025, 2, 31))

print(f"START_DATE: {START_DATE}")
print(f"END_DATE: {END_DATE}")

async def get_post_author(client, channel_username, message_id):
    try:
        print(f"Fetching author for message {message_id} from channel {channel_username}")
        post = await client.get_messages(channel_username, ids=message_id)
        if post.sender:
            author = await client.get_entity(post.sender)
            return f'@{author.username}' if author.username else "Unknown"
    except Exception as e:
        print(f"Error fetching author: {e}")
    return "Unknown"

def get_message_reactions(reactions):
    if reactions:
        return [{"emoji": reaction.reaction.emoticon, "count": reaction.count} for reaction in reactions.results]
    return []

async def fetch_channel_messages(client, channel):
    all_data = []
    try:
        print(f"Fetching messages from channel: {channel}")
        async for message in client.iter_messages(
            channel, offset_date=START_DATE, reverse=True
        ):
            if message.date <= END_DATE and message.message and message.message.strip():
                print(f"Processing message ID: {message.id} from {message.date}")
                reactions = get_message_reactions(message.reactions)
                author = await get_post_author(client, channel, message.id)
                data = {
                    "channel_name": channel,
                    "post_author": author,
                    "post_date": message.date.strftime(DATE_FORMAT),
                    "message": message.message,
                    "reactions": reactions
                }
                all_data.append(data)
        print(f"Total messages fetched from {channel}: {len(all_data)}")
    except Exception as e:
        print(f"An error occurred while processing channel '{channel}': {e}")
    return all_data

async def main():
    print("Starting the Telegram client...")
    client = TelegramClient('test_crawl', API_ID, API_HASH)
    await client.start()

    for channel in TELEGRAM_CHANNELS:
        print(f"\nProcessing channel: {channel}")
        all_data = await fetch_channel_messages(client, channel)

        if all_data:
            output_file = f'{channel}_2024_data.csv'
            try:
                print(f"Writing data to CSV file: {output_file}")
                with open(output_file, 'w', newline='', encoding='utf-8') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow(all_data[0].keys())
                    for item in all_data:
                        csv_writer.writerow(item.values())
                print(f"Data for channel '{channel}' saved to '{output_file}'")
            except Exception as e:
                print(f"Error saving data for channel '{channel}': {e}")
        else:
            print(f"No data found for channel '{channel}'.")

if __name__ == '__main__':
    print("Starting the script...")
    asyncio.run(main())
    print("Script finished.")
