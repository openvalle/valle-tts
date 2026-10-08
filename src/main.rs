use anyhow::{Result, ensure};
use clap::{Args, Parser, Subcommand};
use std::path::PathBuf;
use valle_tts::{
    Language, ReferenceVoice, SynthesisOptions, TtsEngine, WavOutput,
    cache::{ModelCache, builtin_model, builtin_models},
    models::qwen3::Qwen3,
};

#[derive(Parser)]
#[command(version, about = "Local Chinese/English TTS and voice cloning")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    /// List revision-pinned downloadable models.
    Models,
    /// Download or verify the selected model and its codec.
    Download(CacheArgs),
    /// Clone a voice and stream synthesized speech to a WAV file.
    Synthesize(Box<SynthesizeArgs>),
}
#[derive(Args)]
struct SynthesizeArgs {
    #[command(flatten)]
    cache: CacheArgs,
    #[arg(
        long,
        required_unless_present = "text_file",
        conflicts_with = "text_file"
    )]
    text: Option<String>,
    #[arg(long)]
    text_file: Option<PathBuf>,
    #[arg(long)]
    reference: PathBuf,
    #[arg(long,required_unless_present_any=["ref_text_file","x_vector_only"],conflicts_with_all=["ref_text_file","x_vector_only"])]
    ref_text: Option<String>,
    #[arg(long, conflicts_with = "x_vector_only")]
    ref_text_file: Option<PathBuf>,
    #[arg(long)]
    x_vector_only: bool,
    #[arg(long, default_value = "auto")]
    language: String,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 42)]
    seed: i64,
    #[arg(long, default_value_t = 512)]
    max_tokens: u32,
    #[arg(long, default_value_t = 160)]
    max_chunk_chars: usize,
    #[arg(long, default_value_t = 0.9)]
    temperature: f32,
}
#[derive(Args)]
struct CacheArgs {
    #[arg(long, default_value = "qwen3-tts-0.6b-base-q8")]
    model: String,
    #[arg(long)]
    cache_dir: Option<PathBuf>,
    #[arg(long)]
    offline: bool,
}
fn resolve(args: &CacheArgs) -> Result<PathBuf> {
    let root = args
        .cache_dir
        .clone()
        .map(Ok)
        .unwrap_or_else(ModelCache::default_path)?;
    ModelCache::new(root).ensure(&builtin_model(&args.model)?, args.offline)
}
fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Models => println!("{}",serde_json::to_string_pretty(&builtin_models().iter().map(|m|serde_json::json!({"id":m.id,"repository":m.repository,"revision":m.revision,"license":m.license,"bytes":m.files.iter().map(|f|f.bytes).sum::<u64>()})).collect::<Vec<_>>())?),
        Command::Download(args) => println!("{}",resolve(&args)?.display()),
        Command::Synthesize(args) => {
            let SynthesizeArgs { cache,text,text_file,reference,ref_text,ref_text_file,x_vector_only,language,output,seed,max_tokens,max_chunk_chars,temperature } = *args;
            let text = match (text,text_file) { (Some(t),_)=>t,(_,Some(p))=>std::fs::read_to_string(p)?,_=>anyhow::bail!("provide --text or --text-file") };
            ensure!(!text.trim().is_empty(),"text is empty");
            let transcript = if x_vector_only { None } else { match (ref_text,ref_text_file) {
                (Some(t),_)=>Some(t),(_,Some(p))=>Some(std::fs::read_to_string(p)?.trim().to_owned()),_=>None } };
            let voice = ReferenceVoice::from_wav(reference,transcript)?;
            let options = SynthesisOptions { language:Language::parse(&language)?,seed,max_tokens,max_chunk_chars,temperature,..Default::default() };
            let dir = resolve(&cache)?;
            let spec = builtin_model(&cache.model)?;
            let backend = Qwen3::load(&cache.model,dir.join(&spec.files[0].path),dir.join(&spec.files[1].path))?;
            let mut engine = TtsEngine::new(); engine.register(backend)?;
            let mut writer = WavOutput::new(output,24000)?;
            let summary = engine.synthesize(&cache.model,&text,&voice,&options,&mut |chunk|writer.write(chunk))?;
            writer.finish()?;
            println!("{}",serde_json::to_string_pretty(&summary)?);
        }
    }
    Ok(())
}
