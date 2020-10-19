from gradients_visualization.command import OptimizationCommand

exit_command = False

print()

while not exit_command:
    command = input("> ")

    if command.strip() == "exit":
        break

    cmd = OptimizationCommand.from_string(command)
    cmd.run()

# TODO: Add mode with predefined functions
