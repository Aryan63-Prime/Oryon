package com.oryon.ai

interface TokenCallback {
    fun onToken(token: String)
}

object OryonEngine {
    init {
        System.loadLibrary("oryon_lm")
    }

    external fun nativeProbeHardware(nativeLibDir: String): String

    external fun nativeLoadModel(
        modelPath: String,
        tokenizerPath: String,
        nativeLibDir: String,
        maxCtx: Int
    ): Long

    external fun nativeModelInfo(handle: Long): String?

    external fun nativeGenerate(
        handle: Long,
        prompt: String,
        maxTokens: Int,
        threads: Int,
        temperature: Float,
        minP: Float,
        callback: TokenCallback
    ): String?

    external fun nativeRelease(handle: Long)
}
