# terminal2F

REBORN

### TO-DOS

# QueueHandler/QueueListener + Rerun can get weird on quit.

# When putting logging behind a queue, the actual send to Rerun happens on the listener thread,

# and Rerun’s flush guarantees are basically “for the calling thread” (other threads can still

# have stuff in flight). So it can look like the queue handler kinda fucks with the Rerun logs

# at shutdown (last messages missing / not fully flushed).

# Keeping it simple (no queue) for now, but this is something to look into later

# if/when we re-add the queue (likely needs an explicit shutdown order: stop listener -> flush/disconnect).
