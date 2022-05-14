def main(unused_argv):
    gin.parse_config_file(FLAGS.config)
    trainer = Trainer(TrainerConfig())
    trainer.train_and_evaluate()


if __name__ == "__main__":
  app.run(main)