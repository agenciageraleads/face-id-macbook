# face-id-macbook

Unlock your Mac by looking at it: on-device face recognition at the lock screen,
hardened against photos and videos. Free, offline, GPL-3.0.

This project is a modified version of [FaceMac](https://github.com/c1osed1/FaceMac)
by c1osed1. See [What changed](#what-changed-from-facemac) and [NOTICE.md](NOTICE.md).
The app itself is still called FaceMac.

Not affiliated with or endorsed by Apple. "Face ID" and "MacBook" are trademarks of
Apple Inc. and are used here only to describe what the project does.

## How it works

Move the mouse or press a key at the lock screen, or open the lid. The notch plays a
scan animation, then a checkmark, and you are in. If it fails, the notch shakes and
you log in with your password as usual.

```
camera frame
  → Vision face detection + landmarks
  → SFace CoreML embedding, compared with your enrolment
  → blink / micro-motion check
  → anti-spoofing score on every frame (MiniFASNet)
  → 5 consecutive frames must agree
  → screen confirmed locked
  → the Keychain password is typed with CGEvent
```

Frames are never written to disk or sent anywhere. The app has no network code; the
internet is only used once, to download models and dependencies at build time.

## Security

**This is a convenience, not a security upgrade. It is not Face ID.** Apple's Face ID
uses a depth sensor. A regular camera only sees a flat image, so no software check
here can be a guarantee.

- **It stores your macOS login password** in the Keychain, because that is how it
  unlocks. If that risk is not acceptable on your Mac, do not install it.
- **Photo and video detection is limited.** It was measured by one person, on one
  MacBook Air M4, in one environment, with about 20 attempts. Real faces scored mostly
  above 0.79 and images on a phone at most 0.57, which set the cutoff at 0.7. The attack
  tests ran with a cutoff of 0.5; **the final 0.7 cutoff was not attacked again**.
  Printed photos, large monitors and masks were never tested. Expect a different result
  on your camera.
- **Turning off liveness in Settings also turns off this check.**
- **Any local malware running as you** can replace the enrolment file in
  `~/Library/Application Support/FaceMac/`.
- **Anyone who wakes your Mac while you are in front of the camera can unlock it.**
- **After a restart you always need your password.** With FileVault the disk stays
  locked until then. The app only works at the lock screen of a running session, so
  add it to your Login Items.
- It cannot lock you out: if it fails or you quit it, you log in as before.

## Install

There are no prebuilt binaries. You build it yourself, with your own signature.

Requirements: Apple silicon Mac, macOS 14 or later, Xcode, a free Apple Development
certificate, Python 3.12+ and `brew install xcodegen`.

```sh
git clone https://github.com/agenciageraleads/face-id-macbook.git
cd face-id-macbook
scripts/fetch-model.sh
DEVELOPMENT_TEAM=XXXXXXXXXX scripts/install.sh
```

Then, from the menu bar icon:

1. **Set Saved Password**: your macOS login password.
2. **Grant Accessibility**: so it can type at the lock screen.
3. **Enroll Face**: five quick head poses.
4. Add the app to **System Settings → General → Login Items**.

## Uninstall

1. In the menu bar icon, remove the saved password and the enrolled face, then quit.
2. Delete `~/Applications/FaceMac.app` and `~/Library/Application Support/FaceMac/`.
3. In System Settings, remove it from Login Items, Accessibility and Camera.

## What changed from FaceMac

- Added photo and video detection (Silent-Face-Anti-Spoofing), failing closed.
- Requires 5 consecutive frames instead of 3.
- The password is only typed when the screen is confirmed locked. Before, a scan on an
  unlocked screen could type it into the frontmost app, followed by Return. The "type
  password now" action was removed.
- Fixed the camera turning on with the lid closed during dark wakes.
- Hardened Runtime on, and no debug entitlement in the installed build.
- Scans start on mouse or keyboard activity, with no button on the lock screen.
- Notch shows the animation only, with no text or match percentage.
- Dependencies and model downloads pinned where possible, with checksums.
- Removed: Homebrew cask and release job (they shipped the unhardened build), Russian
  and Chinese READMEs, and old screenshots.

## Build and test

```sh
swift build
swift test
xcodegen generate
```

## License

GPL-3.0, like the original. See [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md).
