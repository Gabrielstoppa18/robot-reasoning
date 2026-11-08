# Analise principal (Secao 6.5): modelo logistico misto para o sucesso fisico S,
# com a INSTRUCAO como unidade de generalizacao (efeitos aleatorios de instrucao
# e de semente aninhada na instrucao; Bates et al., 2015).
#
# Entrada: episodes.csv gerado por scripts/bench_analyze.py
# Uso:     Rscript benchmark/analysis/glmm.R benchmark/results/principal-v1/episodes.csv
# Pacotes: install.packages(c("lme4", "emmeans"))
#
# Os modelos abaixo sao os PRE-REGISTRADOS (PROTOCOLO.md, secao 8). Qualquer
# modelo adicional e exploratorio e deve ser relatado como tal.

suppressPackageStartupMessages({
  library(lme4)
  library(emmeans)
})

args <- commandArgs(trailingOnly = TRUE)
path <- if (length(args) > 0) args[1] else "episodes.csv"
d <- read.csv(path, stringsAsFactors = FALSE)
d <- subset(d, rep == 0 & provider != "baseline")
d$model <- factor(d$model)
d$cell <- interaction(d$task, d$seed, drop = TRUE)  # semente aninhada na instrucao
ctrl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
# Com um so LLM no estudo, os termos de modelo saem das formulas (H1, H2 e H5
# viram efeitos do proprio modelo); comparacoes entre modelos sao puladas.
multi <- nlevels(d$model) > 1
fx <- function(rhs_multi, rhs_single) as.formula(paste("S ~", if (multi) rhs_multi else rhs_single))

report <- function(title, fit) {
  cat("\n==========", title, "==========\n")
  print(summary(fit)$coefficients)
  vc <- as.data.frame(VarCorr(fit))
  cat("\nComponentes de variancia (escala logit):\n")
  print(vc[, c("grp", "vcov", "sdcor")])
  # ICC latente: fracao da variancia atribuida a instrucao (pi^2/3 = residuo logistico)
  v_task <- sum(vc$vcov[vc$grp == "task"])
  cat(sprintf("ICC(instrucao) latente = %.3f\n", v_task / (sum(vc$vcov) + pi^2 / 3)))
}

# ---- H1: custo semantico (indireta vs gemea direta), guiada, F1-F6 ----------
h1 <- subset(d, condition == "guided" & expected == "act" & task %in% unique(d$task[d$variant == "direct"]))
h1$indirect <- as.integer(h1$variant == "indirect")
if (nrow(h1) > 0) {
  f1 <- glmer(fx("model * indirect + (1 + indirect | task) + (1 | cell)", "indirect + (1 + indirect | task) + (1 | cell)"),
              data = h1, family = binomial, control = ctrl)
  report("H1: S ~ (modelo x) indireta", f1)
  cat("\nCusto semantico (razao de chances direta/indireta):\n")
  print(contrast(emmeans(f1, if (multi) ~ indirect | model else ~ indirect), "revpairwise", type = "response"))
}

# ---- H2: scaffolding (guiada vs minima), indireta, F1-F6 ----------------------
h2 <- subset(d, condition %in% c("guided", "minimal") & variant == "indirect" & expected == "act")
h2$minimal <- as.integer(h2$scaffolding == "minimal")
if (nrow(h2) > 0) {
  f2 <- glmer(fx("model * minimal + (1 | task) + (1 | cell)", "minimal + (1 | task) + (1 | cell)"),
              data = h2, family = binomial, control = ctrl)
  report("H2: S ~ (modelo x) minima", f2)
  print(contrast(emmeans(f2, if (multi) ~ minimal | model else ~ minimal), "revpairwise", type = "response"))
}

# ---- H5: forma superficial (parafrase, cor do objeto movido, canonica) ---------
h5 <- subset(d, condition == "guided" & variant == "indirect" & expected == "act")
if (nrow(h5) > 0) {
  # eixo da parafrase (base / lexical / contexto); sem rotulo, cai no indice
  h5$paraphrase <- factor(ifelse(is.na(h5$paraphrase_axis) | h5$paraphrase_axis == "",
                                 paste0("p", h5$paraphrase), h5$paraphrase_axis))
  h5$moved_color <- factor(ifelse(h5$moved_color == "", NA, h5$moved_color))
  f5a <- glmer(fx("model + (1 | task) + (1 | cell)", "1 + (1 | task) + (1 | cell)"),
               data = h5, family = binomial, control = ctrl)
  f5b <- glmer(fx("model + paraphrase + moved_color + canonical + (1 | task) + (1 | cell)",
                  "paraphrase + moved_color + canonical + (1 | task) + (1 | cell)"),
               data = subset(h5, !is.na(moved_color)), family = binomial, control = ctrl)
  report("H5: modelo base (variancia de instrucao)", f5a)
  report("H5: com parafrase, cor e canonica", f5b)
  f5a2 <- update(f5a, data = subset(h5, !is.na(moved_color)))
  cat("\nTeste de razao de verossimilhanca (invariancia a forma superficial):\n")
  print(anova(f5a2, f5b))
}

# ---- H3: F7 -- agir indevidamente, por modelo (Fisher exato par a par, Holm) ------
h3 <- subset(d, family == "F7" & condition %in% c("guided", "minimal"))
pairwise_fisher <- function(df, outcome) {
  ms <- sort(unique(df$model))
  out <- data.frame()
  for (i in seq_along(ms)) for (j in seq_along(ms)) if (i < j) {
    sub <- subset(df, model %in% c(ms[i], ms[j]))
    tab <- table(droplevels(factor(sub$model)), factor(sub[[outcome]], levels = c(0, 1)))
    out <- rbind(out, data.frame(a = ms[i], b = ms[j], p = fisher.test(tab)$p.value))
  }
  if (nrow(out) > 0) out$p_holm <- p.adjust(out$p, "holm")
  out
}
if (nrow(h3) > 0) {
  h3$acted <- 1 - h3$nothing_moved
  cat("\n========== H3: proporcao que agiu em F7 ==========\n")
  print(aggregate(cbind(acted, S) ~ model, data = h3, FUN = mean))
  if (multi) print(pairwise_fisher(h3, "acted"))
}

# ---- H4: perturbacao -- escolha de place_on e S por primitiva ----------------------
h4 <- subset(d, condition == "perturb")
if (nrow(h4) > 0) {
  h4$chose_place_on <- as.integer(h4$uses_place_on > 0)
  cat("\n========== H4: escolha de place_on e sucesso ==========\n")
  print(aggregate(cbind(chose_place_on, S) ~ model, data = h4, FUN = mean))
  if (length(unique(h4$model)) > 1) print(pairwise_fisher(h4, "chose_place_on"))
  if (length(unique(h4$chose_place_on)) > 1) {
    f4 <- glmer(S ~ chose_place_on + (1 | task), data = h4, family = binomial, control = ctrl)
    report("H4: S ~ usou place_on", f4)
  }
}

# ---- Comparacao entre modelos e familias (guiada, indireta) ----------------------
if (nrow(h5) > 0 && length(unique(h5$model)) > 1) {
  f6 <- glmer(S ~ model + family + (1 | task) + (1 | cell), data = h5, family = binomial, control = ctrl)
  report("Modelos e familias", f6)
  cat("\nContrastes entre modelos (Holm):\n")
  print(pairs(emmeans(f6, ~ model), adjust = "holm", type = "response"))
}
