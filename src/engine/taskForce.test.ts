// @ts-nocheck
import { Agent } from "../agents/agent";
import { ExecutionMode } from "../configs/enum";
import { Task } from "../tasks/task";
import { TaskForce } from "./taskForce";
import * as delegationGuard from "../agents/delegation.guard";
import OpenAI from "openai";

jest.mock("openai", () => {
  const mockOpenAI = jest.fn().mockImplementation(() => ({
    chat: {
      completions: {
        create: jest
          .fn()
          .mockResolvedValue({ choices: [{ message: { content: "{}" } }] }),
      },
    },
  }));
  return {
    __esModule: true,
    default: mockOpenAI,
    OpenAI: mockOpenAI,
  };
});

jest.mock("../agents/smartManagerAgent", () => {
  return {
    SmartManagerAgent: jest.fn().mockImplementation(() => ({
      setVerbose: jest.fn(),
      planTasks: jest.fn((tasks) => tasks),
      decomposeTask: jest.fn((mainTask, agents, verbose) => [mainTask]),
      assignAgent: jest.fn(),
      evaluateTaskOutput: jest.fn((task, finalOutput) => {
        if (finalOutput.startsWith("DELEGATE")) {
          return { action: "accept" };
        }
        return { action: "accept" };
      }),
      reviewFinalOutput: jest.fn(() => ({ action: "accept" })),
    })),
  };
});

class DummyAgent extends Agent {
  async runTask(...args: any[]) {
    return "Dummy Output";
  }
}

const dummyTask = new Task({
  id: "t1",
  name: "Test Task",
  description: "Test the TaskForce run",
  agent: "Dummy Agent",
  outputFormat: "text",
});

describe("TaskForce", () => {
  it("should run a single dummy task sequentially", async () => {
    const dummyAgent = new DummyAgent({
      name: "Dummy Agent",
      role: "Test",
      goal: "Testing",
      model: "gpt-4o-mini",
      backstory: "This is a dummy agent for testing.",
    });

    const tf = new TaskForce({
      agents: [dummyAgent],
      tasks: [dummyTask],
      executionMode: ExecutionMode.Sequential,
      verbose: false,
    });

    const { result, executedTaskIds } = await tf.run({ input: "test" });

    expect(result).toEqual({ t1: "Dummy Output" });
    expect(executedTaskIds).toEqual(["t1"]);
  });

  it("should trigger weak delegation score and retry", async () => {
    const runTaskMock = jest
      .fn()
      .mockResolvedValueOnce('DELEGATE(X, "reason")')
      .mockResolvedValueOnce("Delegation handled.");

    class WeakDelegationAgent extends Agent {
      async runTask() {
        return runTaskMock();
      }
    }

    jest
      .spyOn(delegationGuard, "checkDelegationScore")
      .mockImplementation((output) => {
        if (output.startsWith("DELEGATE")) {
          return { isWeak: true, reason: "Weak delegation", score: 3 };
        }
        return { isWeak: false, score: 10 };
      });

    const weakAgent = new WeakDelegationAgent({
      name: "WeakAgent",
      role: "Test",
      goal: "Testing",
      model: "gpt-4o-mini",
      backstory: "Weak delegation agent.",
    });

    const task = new Task({
      id: "tweak",
      name: "Test Weak Delegation",
      description: "Test weak delegation scoring.",
      agent: "WeakAgent",
      outputFormat: "text",
    });

    const tf = new TaskForce({
      agents: [weakAgent],
      tasks: [task],
      executionMode: ExecutionMode.Hierarchical,
      verbose: false,
    });

    tf.managerAgent.evaluateTaskOutput = jest.fn((task, finalOutput) => {
      if (finalOutput.startsWith("DELEGATE")) {
        return { action: "accept", forceAccept: true };
      }
      return { action: "accept" };
    });

    const { result } = await tf.run({});

    expect(result.tweak).toBe("Delegation handled.");
    expect(runTaskMock).toHaveBeenCalledTimes(2);
  });

  it("should return correct agent by name", () => {
    const dummyAgent = new DummyAgent({
      name: "Dummy Agent",
      role: "Test",
      goal: "Testing",
      model: "gpt-4o-mini",
      backstory: "This is a dummy agent for testing.",
    });
    const tf = new TaskForce({
      agents: [dummyAgent],
      tasks: [dummyTask],
      executionMode: ExecutionMode.Sequential,
      verbose: false,
    });
    expect(tf.getAgentByName("Dummy Agent")).toBe(dummyAgent);
    expect(tf.getAgentByName("Y")).toBeUndefined();
  });

  it("should list all agent names", () => {
    const agents = [
      new DummyAgent({
        name: "Dummy Agent 1",
        role: "Test",
        goal: "Testing",
        model: "gpt-4o-mini",
        backstory: "This is a dummy agent for testing.",
      }),
      new DummyAgent({
        name: "Dummy Agent 2",
        role: "Test",
        goal: "Testing",
        model: "gpt-4o-mini",
        backstory: "This is a dummy agent for testing.",
      }),
    ];
    const tf = new TaskForce({
      agents: agents,
      tasks: [dummyTask],
      executionMode: ExecutionMode.Sequential,
      verbose: false,
    });
    expect(tf.listAgentNames()).toEqual(["Dummy Agent 1", "Dummy Agent 2"]);
  });

  it("should throw error if agent not found", async () => {
    const tf = new TaskForce({
      agents: [],
      tasks: [dummyTask],
      executionMode: ExecutionMode.Sequential,
    });
    await expect(tf.run({ input: "test" })).resolves.toBeDefined();
    // veya throw etmesini bekliyorsan .rejects.toThrow
  });

  it.todo("should replan and then stop at replan limit");

  it.todo("should use tool handler if TOOL() pattern present in output");
});
