## Installation:

This app requires the following apt packages:
- `python3-tk`
- `xterm`
- `sshfs`
- `sshpass`

For running this app, you need to have configured XTerm and sshfs:
- XTerm: in the file `~/.screenrc`, write: `termcapinfo xterm* ti@:te@`
- sshfs: in the file `/etc/fuse.conf`, with root permission, write: `user_allow_other`