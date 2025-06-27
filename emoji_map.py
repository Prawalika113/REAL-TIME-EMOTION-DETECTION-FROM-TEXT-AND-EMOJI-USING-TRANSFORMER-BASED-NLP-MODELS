emoji_map = {
    '😊': 'happy', '😢': 'sad', '😠': 'angry', '😍': 'in love', '😭': 'crying',
    '😬': 'nervous', '😏': 'smug', '😇': 'angelic', '😋': 'delicious',
    '🥰': 'love',
    '😡': 'furious', '😱': 'surprised', '😨': 'scared', '😂': 'laughing',
    '❤️': 'love', '👍': 'good', '😞': 'disappointed', '😔': 'upset',
    '😁': 'excited', '😄': 'joyful', '😃': 'cheerful', '🙂': 'content',
    '🙁': 'unhappy', '😟': 'worried', '😖': 'frustrated', '😩': 'exhausted',
    '😤': 'annoyed', '😪': 'sleepy', '😴': 'tired', '🤒': 'sick',
    '🤕': 'injured', '😷': 'unwell', '🤧': 'sneezing', '🥵': 'hot',
    '🥶': 'cold', '🤢': 'disgusted', '🤮': 'vomiting', '🥴': 'woozy',
    '😵': 'dizzy', '🤯': 'mind blown', '😬': 'awkward', '🙄': 'eyeroll',
    '😳': 'embarrassed', '🥺': 'pleading', '😇': 'innocent', '🤠': 'playful',
    '😎': 'cool', '🤓': 'nerdy', '🧐': 'curious', '😕': 'confused',
    '😶': 'speechless', '😐': 'neutral', '😑': 'blank', '😒': 'unamused',
    '😈': 'mischievous', '👿': 'evil', '🤬': 'cursing', '💀': 'dead',
    '👻': 'ghost', '💩': 'poop', '🤡': 'clown', '👽': 'alien',
    '🤖': 'robot', '🎃': 'halloween', '🫠': 'melting', '🫥': 'invisible',
    '😺': 'smiling cat', '😸': 'happy cat', '😹': 'laughing cat',
    '😻': 'loving cat', '😼': 'smirking cat', '🙀': 'scared cat',
    '😽': 'kissing cat', '😾': 'angry cat', '👋': 'hello', '🤝': 'handshake',
    '🙏': 'pray', '👏': 'applause', '🙌': 'celebrate', '👐': 'open hands',
    '💪': 'strong', '👀': 'look', '💔': 'heartbroken', '🔥': 'fire',
    '🌈': 'rainbow', '⭐': 'star', '🌟': 'shining', '✨': 'sparkles',
    '☀️': 'sun', '🌙': 'moon', '💤': 'sleeping', '🎉': 'party',
    '🎂': 'birthday', '🍰': 'cake', '🍕': 'pizza', '🍔': 'burger',
    '🍟': 'fries', '🍩': 'donut', '☕': 'coffee', '🍺': 'beer',
    '🚗': 'car', '✈️': 'airplane', '🏠': 'home', '📚': 'books',
    '💻': 'laptop', '📱': 'phone', '📷': 'camera', '🎧': 'music',
    '🎮': 'gaming', '⚽': 'football', '🏀': 'basketball', '🎾': 'tennis',
    '🥇': 'winner', '🏆': 'trophy', '💣': 'bomb', '🔒': 'lock',
    '🔑': 'key', '❤️‍🔥': 'burning love', '💘': 'love struck', '💞': 'loving',
    '💓': 'heartbeat', '💗': 'growing heart', '💖': 'sparkling heart',
    # Happy / Positive Emotions
    '😊': 'happy', '😃': 'cheerful', '😄': 'joyful', '😁': 'excited',
    '🙂': 'content', '😎': 'cool', '😸': 'happy cat', '😺': 'smiling cat',
    '😻': 'loving cat', '😽': 'kissing cat', '😹': 'laughing cat',
    '🥰': 'loving', '😍': 'in love', '🤩': 'starstruck', '😘': 'kiss',
    '😙': 'smiling kiss', '😚': 'blushing kiss', '🤗': 'hugging',
    '🎉': 'celebration', '🙌': 'celebrating', '👏': 'applause', '💃': 'dancing',
    '🕺': 'dancing', '🎶': 'music', '🎵': 'song', '🎂': 'birthday',
    '🥳': 'party', '✨': 'sparkles', '🌟': 'shining', '🌈': 'rainbow',

    # Laughter / Humor
    '😂': 'laughing', '🤣': 'hysterical', '😹': 'laughing cat', '😆': 'laughing hard',

    # Sadness / Crying
    '😢': 'sad', '😭': 'crying', '😿': 'crying cat', '🙁': 'unhappy',
    '☹️': 'frown', '😞': 'disappointed', '😔': 'upset', '😟': 'worried',
    '😥': 'relieved tears', '😓': 'teary', '😩': 'tired', '😫': 'exhausted',

    # Anger / Frustration
    '😠': 'angry', '😡': 'furious', '🤬': 'cursing', '😤': 'annoyed',
    '😾': 'angry cat', '💢': 'rage', '😒': 'unamused',

    # Fear / Shock
    '😱': 'surprised', '😨': 'scared', '😰': 'nervous', '😖': 'anxious',
    '😵': 'dizzy', '🥶': 'frozen', '🥴': 'woozy', '🤯': 'mind blown',

    # Disgust / Dislike
    '🤢': 'disgusted', '🤮': 'vomiting', '🙄': 'eyeroll', '😬': 'awkward',

    # Love & Affection
    '❤️': 'love', '💖': 'sparkling heart', '💘': 'arrow heart', '💗': 'growing heart',
    '💓': 'heartbeat', '💞': 'revolving hearts', '💕': 'twin hearts',
    '💟': 'heart decoration', '❤️‍🔥': 'burning love', '💔': 'heartbroken',

    # Approval / Agreement
    '👍': 'thumbs up', '👌': 'okay', '✌️': 'peace', '🤝': 'handshake',
    '🤜': 'fist bump', '🤛': 'fist bump', '🙏': 'pray', '💪': 'strong',

    # Disapproval / Rejection
    '👎': 'thumbs down', '🙅': 'no way', '🙆': 'yes',

    # Sleep / Boredom
    '😴': 'sleepy', '😪': 'sleepy', '💤': 'sleeping', '🥱': 'yawning',

    # Illness / Pain
    '🤒': 'sick', '🤕': 'injured', '🤧': 'sneezing', '😷': 'mask',
    '🤑': 'money face', '🤠': 'playful', '😇': 'innocent',

    # Random / Creative / Fun
    '🤡': 'clown', '👻': 'ghost', '💩': 'poop', '💀': 'dead',
    '👽': 'alien', '🤖': 'robot', '🎃': 'halloween', '🫠': 'melting',
    '🫥': 'invisible', '🔮': 'mystical', '🌙': 'moon', '☀️': 'sun',

    # Objects or Mixed
    '🎈': 'balloon', '🍕': 'pizza', '🍔': 'burger', '🍟': 'fries',
    '🍩': 'donut', '🍦': 'ice cream', '☕': 'coffee', '🍺': 'beer',
    '📱': 'phone', '💻': 'laptop', '🎧': 'music', '🎮': 'gaming',
    '🏠': 'home', '🚗': 'car', '✈️': 'airplane', '🚌': 'bus',
    '⚽': 'football', '🏀': 'basketball', '🎾': 'tennis', '🏆': 'trophy'
}
