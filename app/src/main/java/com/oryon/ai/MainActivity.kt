package com.oryon.ai

import android.content.Context
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

data class ChatMessage(
    val isUser: Boolean,
    val text: String,
    val stats: String? = null
)

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme(colorScheme = darkColorScheme(
                primary = Color(0xFF38BDF8),
                background = Color(0xFF0B0F19),
                surface = Color(0xFF141B2D),
                onBackground = Color(0xFFE2E8F0),
                onSurface = Color(0xFFE2E8F0)
            )) {
                OryonChatScreen()
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun OryonChatScreen() {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val listState = rememberLazyListState()

    var engineHandle by remember { mutableLongStateOf(0L) }
    var statusText by remember { mutableStateOf("Initializing native Rust engine...") }
    var isGenerating by remember { mutableStateOf(false) }
    var isGgufModel by remember { mutableStateOf(false) }
    var inputText by remember { mutableStateOf("") }
    var threads by remember { mutableIntStateOf(6) }
    var temperature by remember { mutableFloatStateOf(0.6f) }
    var selectedBackend by remember { mutableIntStateOf(0) }
    var hardwareBadge by remember { mutableStateOf("Detecting hardware") }
    var hardwareBadgeColor by remember { mutableStateOf(Color(0xFF38BDF8)) }
    var isLoadingModel by remember { mutableStateOf(false) }
    val messages = remember { mutableStateListOf<ChatMessage>() }

    // Load bundled micro-checkpoint on first launch so app works immediately offline
    LaunchedEffect(Unit) {
        withContext(Dispatchers.IO) {
            val hw = OryonEngine.nativeProbeHardware(context.applicationInfo.nativeLibraryDir)
            val modelFile = copyAssetToFiles(context, "oryon_micro_q8.bin")
            val tokFile = copyAssetToFiles(context, "tokenizer.bin")
            val h = OryonEngine.nativeLoadModel(
                modelFile.absolutePath,
                tokFile.absolutePath,
                context.applicationInfo.nativeLibraryDir,
                256
            )
            withContext(Dispatchers.Main) {
                engineHandle = h
                statusText = if (h != 0L) {
                    OryonEngine.nativeModelInfo(h) ?: hw
                } else {
                    "Ready ($hw) — Tap 'Load GGUF' to pick a model"
                }
                if (h != 0L) {
                    val active = OryonEngine.nativeSetBackend(h, 0)
                    hardwareBadge = active
                    hardwareBadgeColor = when {
                        active.startsWith("GPU:", ignoreCase = true) -> Color(0xFF22C55E)
                        active.startsWith("NPU:", ignoreCase = true) -> Color(0xFFA855F7)
                        else -> Color(0xFF38BDF8)
                    }
                }
            }
        }
    }

    DisposableEffect(Unit) {
        onDispose {
            if (engineHandle != 0L) {
                OryonEngine.nativeRelease(engineHandle)
            }
        }
    }

    // Android Storage Access Framework picker for any .gguf model (e.g., Qwen2.5-3B-Q4_0.gguf)
    val modelPicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { uri: Uri? ->
        if (uri != null) {
            scope.launch(Dispatchers.IO) {
                isLoadingModel = true
                withContext(Dispatchers.Main) {
                    statusText = "Copying & loading GGUF model..."
                }
                val dst = File(context.filesDir, "imported_model.gguf")
                val metadata = context.contentResolver.openAssetFileDescriptor(uri, "r")
                val selectedSize = metadata?.use { it.length } ?: -1L
                val selectedName = uri.lastPathSegment ?: "imported_model.gguf"
                val cacheNameFile = File(context.filesDir, "imported_model.name")
                val cacheMatches = dst.exists() && dst.length() == selectedSize &&
                    cacheNameFile.exists() && cacheNameFile.readText() == selectedName
                if (!cacheMatches) {
                    context.contentResolver.openInputStream(uri)?.use { input ->
                        dst.outputStream().use { output -> input.copyTo(output) }
                    }
                    cacheNameFile.writeText(selectedName)
                }
                if (engineHandle != 0L) {
                    OryonEngine.nativeRelease(engineHandle)
                }
                val h = OryonEngine.nativeLoadModel(
                    dst.absolutePath,
                    "",
                    context.applicationInfo.nativeLibraryDir,
                    8192
                )
                withContext(Dispatchers.Main) {
                    engineHandle = h
                    isGgufModel = (h != 0L)
                    isLoadingModel = false
                    statusText = if (h != 0L) {
                        OryonEngine.nativeModelInfo(h) ?: "GGUF loaded"
                    } else {
                        "Failed to load GGUF (use Q4_0, Q8_0, F16, or F32 GGUF)"
                    }
                }
            }
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Column {
                        Text("Oryon AI", fontWeight = FontWeight.Bold, fontSize = 18.sp)
                        Text(
                            statusText,
                            fontSize = 11.sp,
                            color = Color(0xFF94A3B8),
                            maxLines = 2
                        )
                    }
                },
                actions = {
                    TextButton(onClick = { messages.clear(); inputText = "" }) {
                        Text("Clear Chat")
                    }
                    TextButton(
                        enabled = !isGenerating,
                        onClick = { modelPicker.launch(arrayOf("*/*")) }
                    ) {
                        Text("Load GGUF", fontWeight = FontWeight.SemiBold)
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(containerColor = Color(0xFF111827))
            )
        },
        containerColor = Color(0xFF0B0F19)
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(Color(0xFF111827))
                    .padding(horizontal = 12.dp, vertical = 8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                listOf("CPU", "GPU", "NPU").forEachIndexed { index, label ->
                    val mode = index + 1
                    FilterChip(
                        selected = selectedBackend == mode,
                        onClick = {
                            if (engineHandle != 0L) {
                                scope.launch(Dispatchers.IO) {
                                    val active = OryonEngine.nativeSetBackend(engineHandle, mode)
                                    withContext(Dispatchers.Main) {
                                        selectedBackend = mode
                                        hardwareBadge = active
                                        hardwareBadgeColor = when {
                                            active.startsWith("GPU:", ignoreCase = true) -> Color(0xFF22C55E)
                                            active.startsWith("NPU:", ignoreCase = true) -> Color(0xFFA855F7)
                                            else -> Color(0xFF38BDF8)
                                        }
                                        statusText = active
                                    }
                                }
                            }
                        },
                        label = { Text(label) },
                        modifier = Modifier.padding(end = 8.dp)
                    )
                }
                Spacer(Modifier.weight(1f))
                Box(
                    modifier = Modifier
                        .background(hardwareBadgeColor, RoundedCornerShape(10.dp))
                        .padding(horizontal = 9.dp, vertical = 5.dp)
                ) {
                    Text(hardwareBadge, color = Color.White, fontSize = 10.sp)
                }
            }

            // Compact hardware controls bar
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(Color(0xFF0F172A))
                    .padding(horizontal = 12.dp, vertical = 6.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                if (isLoadingModel) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(18.dp),
                        strokeWidth = 2.dp,
                        color = Color(0xFF38BDF8)
                    )
                    Spacer(Modifier.width(8.dp))
                }
                Text("Threads: $threads", fontSize = 12.sp, color = Color(0xFF94A3B8))
                Slider(
                    value = threads.toFloat(),
                    onValueChange = { threads = it.toInt() },
                    valueRange = 1f..8f,
                    steps = 6,
                    modifier = Modifier.width(120.dp)
                )
                Text(
                    "Temp: ${"%.1f".format(temperature)}",
                    fontSize = 12.sp,
                    color = Color(0xFF94A3B8)
                )
                Slider(
                    value = temperature,
                    onValueChange = { temperature = it },
                    valueRange = 0f..1.2f,
                    modifier = Modifier.width(100.dp)
                )
            }

            // Message stream
            LazyColumn(
                state = listState,
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth()
                    .padding(horizontal = 12.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp),
                contentPadding = PaddingValues(vertical = 12.dp)
            ) {
                items(messages) { msg ->
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = if (msg.isUser) Arrangement.End else Arrangement.Start
                    ) {
                        Column(
                            modifier = Modifier
                                .widthIn(max = 310.dp)
                                .background(
                                    color = if (msg.isUser) Color(0xFF1D4ED8) else Color(0xFF1E293B),
                                    shape = RoundedCornerShape(14.dp)
                                )
                                .padding(12.dp)
                        ) {
                            Text(msg.text, color = Color.White, fontSize = 14.sp)
                            if (msg.stats != null) {
                                Spacer(Modifier.height(4.dp))
                                Text(
                                    msg.stats,
                                    color = Color(0xFF38BDF8),
                                    fontSize = 10.sp,
                                    fontFamily = FontFamily.Monospace
                                )
                            }
                        }
                    }
                }
            }

            // Prompt input bar
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(Color(0xFF111827))
                    .padding(10.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                OutlinedTextField(
                    value = inputText,
                    onValueChange = { inputText = it },
                    modifier = Modifier.weight(1f),
                    placeholder = { Text("Ask locally (offline)...", fontSize = 14.sp) },
                    singleLine = false,
                    maxLines = 3
                )
                Spacer(Modifier.width(8.dp))
                if (isGenerating) {
                    Button(
                        onClick = {
                            if (engineHandle != 0L) {
                                OryonEngine.nativeCancelGeneration(engineHandle)
                            }
                        },
                        colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFDC2626))
                    ) {
                        Text("Stop")
                    }
                } else {
                    Button(
                        enabled = engineHandle != 0L && inputText.isNotBlank(),
                        onClick = {
                            val userPrompt = inputText.trim()
                            inputText = ""
                            messages.add(ChatMessage(isUser = true, text = userPrompt))
                            val replyIdx = messages.size
                            messages.add(ChatMessage(isUser = false, text = ""))
                            isGenerating = true

                            val formatted = if (isGgufModel) {
                                "<|im_start|>user\n$userPrompt<|im_end|>\n<|im_start|>assistant\n"
                            } else {
                                "Q: $userPrompt\nA:"
                            }

                            scope.launch(Dispatchers.IO) {
                                val sb = StringBuilder()
                                val stats = OryonEngine.nativeGenerate(
                                    handle = engineHandle,
                                    prompt = formatted,
                                    maxTokens = 256,
                                    threads = threads,
                                    temperature = temperature,
                                    minP = 0.05f,
                                    callback = object : TokenCallback {
                                        override fun onToken(token: String) {
                                            sb.append(token)
                                            val snapshot = sb.toString()
                                                .removeSuffix("\n\nQ:")
                                                .removeSuffix("<|im_end|>")
                                            scope.launch(Dispatchers.Main) {
                                                messages[replyIdx] = ChatMessage(isUser = false, text = snapshot)
                                            }
                                        }
                                    }
                                )
                                withContext(Dispatchers.Main) {
                                    val finalTxt = sb.toString()
                                        .removeSuffix("\n\nQ:")
                                        .removeSuffix("<|im_end|>")
                                        .trim()
                                    messages[replyIdx] = ChatMessage(
                                        isUser = false,
                                        text = finalTxt.ifEmpty { "(empty output)" },
                                        stats = stats
                                    )
                                    isGenerating = false
                                    listState.animateScrollToItem(messages.size - 1)
                                }
                            }
                        }
                    ) {
                        Text("Send")
                    }
                }
            }
        }
    }
}

private fun copyAssetToFiles(context: Context, name: String): File {
    val outFile = File(context.filesDir, name)
    if (!outFile.exists()) {
        context.assets.open(name).use { input ->
            outFile.outputStream().use { output -> input.copyTo(output) }
        }
    }
    return outFile
}
