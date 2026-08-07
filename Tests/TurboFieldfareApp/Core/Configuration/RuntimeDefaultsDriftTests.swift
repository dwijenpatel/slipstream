import Foundation
import Testing
import TurboFieldfare
import TurboFieldfareDecodeProtocol

@testable import TurboFieldfareAppCore

/// Every surface must resolve to the same runtime tuning defaults.
///
/// `expertCacheSlots` was declared with a literal in six places. Raising it
/// 16 -> 64 on measured evidence updated two, and the other four kept
/// shipping 16 — including the Mac app's persisted settings, so the app
/// ignored the change entirely. The build stayed green the whole time,
/// because nothing compared the surfaces to each other. This is that
/// comparison.
///
/// A failure here means a surface has drifted, NOT that the numbers need
/// updating. Change `RuntimeDefaults` and let this test confirm the change
/// reached everywhere.
@Suite("Runtime defaults do not drift across surfaces")
struct RuntimeDefaultsDriftTests {
    @Test("The core runtime resolves to RuntimeDefaults")
    func coreMatches() {
        let runtime = RuntimeConfiguration.production
        #expect(runtime.expertCacheSlots == RuntimeDefaults.expertCacheSlots)
        #expect(runtime.prefillChunkTokens == RuntimeDefaults.prefillChunkTokens)
        #expect(runtime.expertCachePolicy == RuntimeDefaults.expertCachePolicy)
        #expect(runtime.rdadvisePolicy == RuntimeDefaults.rdadvisePolicy)
    }

    @Test("The Mac app's options and its PERSISTED settings both match")
    func appMatches() throws {
        let options = AppRuntimeOptions()
        #expect(options.expertCacheSlots == RuntimeDefaults.expertCacheSlots)
        #expect(options.prefillChunkTokens == RuntimeDefaults.prefillChunkTokens)
        #expect(options.prefillEnabled == RuntimeDefaults.prefillEnabled)
        #expect(try options.resolvedRuntimeConfiguration(forceLogitsHead: false)
                == .production)

        // The settings FILE is the one that actually reaches a user: a stored
        // value wins over the option default, which is how the Mac app kept
        // running 16 after the option was changed to 64.
        let stored = MacAppSettings()
        #expect(stored.expertCacheSlots == RuntimeDefaults.expertCacheSlots)
        #expect(stored.prefillEnabled == RuntimeDefaults.prefillEnabled)
    }

    /// The wire format cannot reference RuntimeDefaults — it deliberately has
    /// no dependencies — so its literals are pinned here instead.
    @Test("The decode-service wire format's literals still match")
    func decodeProtocolMatches() {
        let options = DecodeRuntimeOptions()
        #expect(options.expertCacheSlots == RuntimeDefaults.expertCacheSlots)
        #expect(options.prefillChunkTokens == RuntimeDefaults.prefillChunkTokens)
        #expect(options.prefillEnabled == RuntimeDefaults.prefillEnabled)
    }

    /// Bumping the stored-settings version is what migrates existing installs
    /// onto a new default; without it they keep their saved value forever.
    @Test("Stored settings from an older version are rejected and re-defaulted")
    func staleSettingsAreInvalid() {
        var old = MacAppSettings()
        old.version = 1
        #expect(!old.isValid())
        #expect(MacAppSettings().version == MacAppSettings.currentVersion)
    }
}
